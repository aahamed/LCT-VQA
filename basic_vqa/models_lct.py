import torch
import torch.nn as nn
import torchvision.models as models
from pcdarts.model_search import Network
import config
from torch.utils.data import WeightedRandomSampler
from models_base import VqaModel as WModel, softXEnt
from darts.model import NetworkImageNet
from darts.genotypes import DARTS as gtDarts

class ImgEncoder(nn.Module):

    def __init__(self, embed_size, vqa_model, init_ch=16, layers=4):
        """
        Image Encoder using PC-DARTS
        """
        super(ImgEncoder, self).__init__()
        # import pdb; pdb.set_trace()
        self.darts = Network( init_ch, embed_size, layers, vqa_model )


    def forward(self, image):
        """Extract feature vector from image vector.
        """
        # import pdb; pdb.set_trace()
        # image has dimensions [ batch_size, 3, 224, 224 ]
        img_feature = self.darts( image )
        # img_feature has dimensions [ batch_size, embed_size ]
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        # l2 normalized feature vector
        img_feature = img_feature.div(l2_norm)

        return img_feature

class ImgEncoderFixed(nn.Module):

    def __init__(self, embed_size, pretrained=True):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoderFixed, self).__init__()
        # import pdb; pdb.set_trace()
        if config.ARCH_TYPE == 'fixed-vgg':
            model = models.vgg19(pretrained=pretrained)
            in_features = model.classifier[-1].in_features  # input size of feature vector
            model.classifier = nn.Sequential(
                *list(model.classifier.children())[:-1])    # remove last fc layer
        elif config.ARCH_TYPE == 'fixed-darts':
            init_channels = 48
            classes = 1000
            layers = 14
            auxiliary = True
            genotype = gtDarts
            model = NetworkImageNet(init_channels, classes, layers,
                    auxiliary, genotype)
            in_features = ( model.output_size ** 2 ) * model.output_ch
            if pretrained:
                state = torch.load( config.DARTS_MODEL_PATH,
                        map_location=config.DEVICE )
                model.load_state_dict( state['state_dict'] )
        else:
            assert False and f'Unrecognized arch_type: {config.ARCH_TYPE}'

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image
        self.pretrained = pretrained

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        if self.pretrained:
            with torch.no_grad():
                img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        else:
            img_feature = self.model( image )
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature

class NoImgEncoder(nn.Module):

    def __init__(self, embed_size):
        super(NoImgEncoder, self).__init__()
        self.embed_size = embed_size

    def forward(self, image):
        """
        Output 0 embedding
        """
        batch_size = len(image)
        out = torch.zeros(batch_size, self.embed_size).to(
                config.DEVICE)
        return out


class QstEncoder(nn.Module):

    def __init__( self, qst_vocab_size, word_embed_size,
            embed_size, num_layers, hidden_size,
            deterministic=True, temperature=0.1,
            max_length=30 ):

        super(QstEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.deterministic = deterministic
        self.temperature = temperature
        self.max_length = max_length
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states
        self.fc2 = nn.Linear(hidden_size, qst_vocab_size)
        self.softmax = torch.nn.Softmax(dim=2)
        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, question, image_embedding):
        # import pdb; pdb.set_trace()
        self.lstm.flatten_parameters()
        hidden_state = image_embedding.view(1, -1, self.hidden_size )
        # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.word2vec(question)           
        qst_vec = self.tanh(qst_vec)
        # [max_qst_length=30, batch_size, word_embed_size=300]
        qst_vec = qst_vec.transpose(0, 1)                             
        # [num_layers=2, batch_size, hidden_size=512]
        # teacher forcing
        out, (hidden, cell) = self.lstm(qst_vec, 
                (hidden_state, hidden_state))                        
        # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = torch.cat((hidden, cell), 2)                    
        # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     
        # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  
        qst_feature = self.tanh(qst_feature)
        # [batch_size, embed_size]
        qst_feature = self.fc1(qst_feature)                            
        out = out.transpose(0, 1)
        out = self.tanh(out)
        qst_out = self.fc2(out)

        return qst_feature, qst_out

    def generate( self, image_embedding ):
        '''
        Generate question-answer pairs for the given
        image embeddings
        '''
        # import pdb; pdb.set_trace()
        batch_size = len( image_embedding )
        self.lstm.flatten_parameters()
        hidden_state = image_embedding.view(1, -1, self.hidden_size )

        # create start tokens ( numerical value 2 )
        start_word = torch.ones( (batch_size, 1) ).\
                long().to( config.DEVICE ) * 2
        start_vec = self.word2vec( start_word )
        start_vec = self.tanh( start_vec )
        start_vec = start_vec.transpose(0, 1)

        # generate question
        hidden_state = ( hidden_state, hidden_state )
        current_word = start_vec
        qst_out = torch.zeros( (batch_size, self.max_length) )\
                .long().to( config.DEVICE )
        for t in range( self.max_length ):
            out, hidden_state = \
                    self.lstm( current_word, hidden_state )
            out = out.transpose(0, 1)
            out = self.tanh( out )
            prob = self.fc2( out )
            pred = self.sample( prob )
            current_word = self.word2vec( pred )
            current_word = current_word.transpose(0, 1)
            qst_out[:, t] = pred[:, 0]
        
        return qst_out

    def sample( self, prob ):
        pred = None
        if self.deterministic:
            pred = torch.argmax( prob, 2 )
        else:
            soft = self.softmax( prob / self.temperature )
            batch_size = len( prob )
            pred = [ list(
                WeightedRandomSampler(soft[i,0,:], 1)) \
                        for i in range(batch_size)]
            pred = torch.tensor( pred ).to( config.DEVICE )
        return pred



class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size, num_layers,
            hidden_size, pretrained=True):

        super(VqaModel, self).__init__()
        self.img_encoder = None
        if config.NO_IMG_ENC:
            self.img_encoder = NoImgEncoder(embed_size)
        elif config.ARCH_TYPE == 'pcdarts':
            self.img_encoder = ImgEncoder(embed_size, self)
        else:
            self.img_encoder = ImgEncoderFixed(embed_size, pretrained)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size,
                embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.embed_size = embed_size
        self.qst_vocab_size = qst_vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.word_embed_size = word_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, img, qst):

        # import pdb; pdb.set_trace()
        # [batch_size, embed_size]
        img_feature = self.img_encoder(img)                     
        # [batch_size, embed_size]
        qst_feature, qst_out = self.qst_encoder(qst, img_feature)            
        # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # [batch_size, ans_vocab_size=1000]
        combined_feature = self.fc1(combined_feature)           
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        # [batch_size, ans_vocab_size=1000]
        combined_feature = self.fc2(combined_feature)           

        return combined_feature, qst_out

    def generate(self, img):
        '''
        Generate question-answer pairs for the given
        images
        '''
        # import pdb; pdb.set_trace()
        # with torch.no_grad():
        img_feature = self.img_encoder(img)
        # generate question
        qst = self.qst_encoder.generate(img_feature)
        # encode generated questions
        qst_feature, _ = self.qst_encoder(qst, img_feature)
        # get answer for generated question
        combined_feature = torch.mul(img_feature, qst_feature)  
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        answer = self.fc2(combined_feature)           

        return qst, answer


    def genotype( self ):
        return self.img_encoder.darts.genotype()

    def arch_parameters( self ):
        return self.img_encoder.darts.arch_parameters()

    def _loss(self, images, questions, labels):
        ans_out, qst_out = self( images, questions )
        ans_loss = self.criterion( ans_out, labels )
        qst = questions[:, 1:].flatten()
        qst_out = qst_out[:, :-1].flatten(end_dim=1)
        qst_loss = self.criterion( qst_out, qst )
        loss = ans_loss + qst_loss
        return loss

    def new( self ):
        new_darts = self.img_encoder.darts.new()
        new_vqa_model = VqaModel( self.embed_size,
                self.qst_vocab_size, self.ans_vocab_size,
                self.word_embed_size, self.num_layers,
                self.hidden_size )
        new_vqa_model.img_encoder.darts = new_darts
        new_vqa_model.to( config.DEVICE )
        return new_vqa_model

def test_vqa():
    # global config.DEVICE, config.ARCH_TYPE
    config.DEVICE = 'cuda'
    config.ARCH_TYPE = 'pcdarts'
    print( 'Test VQA model' )
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 1000
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    import pdb; pdb.set_trace()
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size ).to(config.DEVICE)
    batch_size = 4
    img_size = 64
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    img = torch.randn( batch_size, 3,
            img_size, img_size ).to(config.DEVICE)
    qst = torch.randint( qst_vocab_size,
            ( batch_size, qst_max_len) ).to(config.DEVICE)
    # test forward pass
    out, qst_out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    assert qst_out.shape == ( batch_size, qst_max_len, qst_vocab_size )
    labels = torch.randint( ans_vocab_size,
            (batch_size, ) ).to(config.DEVICE)
    # test architecture loss
    # loss = model.img_encoder.darts._loss( img, qst, labels )
    new_model = model.new()
    loss = new_model._loss( img, qst, labels )
    # test teacher forcing
    N = batch_size * ( qst_max_len - 1 )
    qst = qst[:, 1:].flatten()
    qst_out = qst_out[:, :-1].flatten( end_dim=1 )
    qst_loss = criterion( qst_out, qst )
    # test qa generation
    qst, ans = model.generate( img )
    assert ans.shape == ( batch_size, ans_vocab_size )
    assert qst.shape == ( batch_size, qst_max_len )
    # test stochastic generation
    model.qst_encoder.deterministic = False
    model.qst_encoder.temperature = 0.01
    qst, ans = model.generate( img )
    assert ans.shape == ( batch_size, ans_vocab_size )
    assert qst.shape == ( batch_size, qst_max_len )
    # test gradient prop for multi-step loss
    w_model = WModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size ).to(config.DEVICE)
    w_ans = w_model( img, qst )
    # argmax not differentiable
    # ans = torch.argmax( ans, 1 )
    # w_loss = criterion( w_ans, ans )
    w_loss = softXEnt( w_ans, ans )
    # The generated qst are obtained via sampling
    # and hence not differentiable. This means gradient
    # will not flow back through the questions which means
    # gradient for model.qst_encoder.fc2 will be None
    grad_model = torch.autograd.grad( w_loss, model.parameters(),
            allow_unused=True )
    print( 'Test passed!' )

def test_vqa_fixed():
    config.DEVICE = 'cuda'
    config.ARCH_TYPE = 'fixed-darts'
    print( 'Test VQA model with fixed encoder' )
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 1000
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    pretrained = True
    import pdb; pdb.set_trace()
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size, pretrained ).to(config.DEVICE)
    batch_size = 4
    img_size = 224
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    img = torch.randn( batch_size, 3,
            img_size, img_size ).to(config.DEVICE)
    qst = torch.randint( qst_vocab_size,
            ( batch_size, qst_max_len) ).to(config.DEVICE)
    # test forward pass
    out, qst_out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    assert qst_out.shape == ( batch_size, qst_max_len, qst_vocab_size )
    labels = torch.randint( ans_vocab_size,
            (batch_size, ) ).to(config.DEVICE)
    loss = model._loss( img, qst, labels )
    loss.backward()
    cnt = 0
    for param in model.img_encoder.parameters():
        if param.grad is None:
            cnt += 1
    print( 'cnt:', cnt )
    print( 'Test passed!' )

def test_vqa_noimg():
    config.DEVICE = 'cuda'
    config.ARCH_TYPE = 'fixed-vgg'
    config.NO_IMG_ENC = True
    print( 'Test VQA model without img encoder' )
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 1000
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    pretrained = True
    import pdb; pdb.set_trace()
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size, pretrained ).to(config.DEVICE)
    batch_size = 4
    img_size = 224
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss().to(config.DEVICE)
    img = torch.randn( batch_size, 3,
            img_size, img_size ).to(config.DEVICE)
    qst = torch.randint( qst_vocab_size,
            ( batch_size, qst_max_len) ).to(config.DEVICE)
    # test forward pass
    out, qst_out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    assert qst_out.shape == ( batch_size, qst_max_len, qst_vocab_size )
    labels = torch.randint( ans_vocab_size,
            (batch_size, ) ).to(config.DEVICE)
    loss = model._loss( img, qst, labels )
    loss.backward()
    config.NO_IMG_ENC = False
    print( 'Test passed!' )

def test():
    test_vqa_noimg()
    test_vqa_fixed()
    test_vqa()

def main():
    test()

if __name__ == '__main__':
    main()
