import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config
from pcdarts.model_search import Network

class VggEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(VggEncoder, self).__init__()
        model = models.vgg19(pretrained=True)
        # input size of feature vector
        in_features = model.classifier[-1].in_features
        # remove last fc layer
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        # loaded model without last fc layer
        self.model = model
        # feature vector of image
        self.fc = nn.Linear(in_features, embed_size)

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            # bs x vgg16(19)_fc=4096
            img_feature = self.model(image)
        # bs x embed_size
        img_feature = self.fc(img_feature)

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        # l2-normalized feature vector
        img_feature = img_feature.div(l2_norm)

        return img_feature

class DartsEncoder(nn.Module):
    '''
    Encode image using darts image encoder
    '''

    def __init__( self, embed_size, init_ch=16, layers=4 ):
        super(DartsEncoder, self).__init__()
        self.darts = Network( init_ch, embed_size, layers )
        in_features = self.darts.output_ch * ( self.darts.output_size ** 2 )
        self.fc = nn.Linear( in_features, embed_size )

    def forward(self, image):
        # img dim: bs x 3 x 224 x 224
        img_feature = self.darts( image )
        img_feature = self.fc( img_feature )
        # img feat dim: bs x embed_size
        l2_norm = img_feature.norm( p=2, dim=1, keepdim=True ).detach()
        # l2 normalized feature vector
        img_feature = img_feature.div(l2_norm)
        return img_feature


def get_img_encoder( img_encoder_type, embed_size ):
    if img_encoder_type == 'vgg':
        return VggEncoder( embed_size )
    elif img_encoder_type == 'darts':
        return DartsEncoder( embed_size )
    else:
        raise Exception( f'Unrecognized encoder type: {img_encoder_type}' )

class QstEncoderBase(nn.Module):
    '''
    Base question encoder
    '''

    def __init__( self, vocab_size, word_embed_size,
            embed_size, num_layers, hidden_size,
            deterministic=True, temperature=0.1,
            max_length=30 ):
        super(QstEncoderBase, self).__init__()
        self.hidden_size = hidden_size
        self.deterministic = deterministic
        self.temperature = temperature
        self.max_length = max_length
        self.word2vec = nn.Embedding(vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, vocab_size)
        self.softmax = torch.nn.Softmax( dim=2 )
        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc1.weight.data)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, qst, image_embedding):
        raise NotImplementedError()

    def generate( self, image_embedding ):
        '''
        Generate questions for the given
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
            prob = self.fc1( out )
            pred = self.sample( prob )
            current_word = self.word2vec( pred )
            current_word = current_word.transpose(0, 1)
            qst_out[:, t] = pred[:, 0]
        
        return qst_out

    def sample(self, prob):
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

class QstEncoder(QstEncoderBase):
    '''
    Question Encoder
    '''

    def __init__(self, qst_vocab_size, word_embed_size,
            embed_size, num_layers, hidden_size,
            deterministic=True, temperature=0.1,
            max_length=30):

        super(QstEncoder, self).__init__(qst_vocab_size, word_embed_size,
                embed_size, num_layers, hidden_size, deterministic,
                temperature, max_length)
        # 2 is for hidden and cell states
        self.fc2 = nn.Linear(2*num_layers*hidden_size, embed_size)
        # weight initialization
        torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, question, image_embedding):

        # import pdb; pdb.set_trace()
        self.lstm.flatten_parameters()
        hidden_state = image_embedding.view(1, -1, self.hidden_size)
        # bs x max_qst_len x word_embed_size
        qst_vec = self.word2vec(question)
        qst_vec = self.tanh(qst_vec)
        # max_qst_len x bs x word_embed_size
        qst_vec = qst_vec.transpose(0, 1)
        # num_layers x bs x hidden_size
        # teacher forcing
        out, (hidden, cell) = self.lstm(qst_vec, (hidden_state, hidden_state))
        # num_layers x bs x num_layers*hidden_size
        qst_feature = torch.cat((hidden, cell), 2)
        # bs x num_layers x 2*hidden_size
        qst_feature = qst_feature.transpose(0, 1)
        # bs x num_layers*2 * hidden_size
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)
        qst_feature = self.tanh(qst_feature)
        # bs x embed_size
        qst_feature = self.fc2(qst_feature)
        out = out.transpose(0, 1)
        out = self.tanh(out)
        qst_out = self.fc1(out)

        return qst_feature, qst_out

class QstAnsEncoder( QstEncoderBase ):
    '''
    Encode question and answer
    '''

    def __init__(self, unified_vocab_size, word_embed_size,
            embed_size, num_layers, hidden_size,
            deterministic=True, temperature=0.1,
            max_length=30):
        super(QstAnsEncoder, self).__init__(unified_vocab_size,
                word_embed_size, embed_size, num_layers, hidden_size,
                deterministic, temperature, max_length)

    def forward(self, qa_str, image_embedding):
        '''
        qa_str: ground truth tokens representing qst <sep> ans
        image_embedding: embedding of image
        returns a vector of shape: bs x max_length x unified_vocab_size
        '''
        # import pdb; pdb.set_trace()
        self.lstm.flatten_parameters()
        hidden_state = image_embedding.view(1, -1, self.hidden_size)
        # bs x max_qst_len x word_embed_size
        qa_vec = self.word2vec(qa_str)
        qa_vec = self.tanh(qa_vec)
        # max_qst_len x bs x word_embed_size
        qa_vec = qa_vec.transpose(0, 1)
        # teacher forcing
        # num_layers x bs x hidden_size
        out, (hidden, cell) = self.lstm(qa_vec, (hidden_state, hidden_state))
        out = out.transpose(0, 1)
        out = self.tanh(out)
        qa_out = self.fc1(out)
        # bs x max_length x unified_vocab_size
        return qa_out


class VqaModelBase(nn.Module):
    '''
    Base VqaModel
    '''
    def __init__(self, embed_size, qst_vocab_size,
        ans_vocab_size, word_embed_size, num_layers,
        hidden_size, img_encoder_type='vgg'):

        super(VqaModelBase, self).__init__()
        self.embed_size = embed_size
        self.qst_vocab_size = qst_vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.word_embed_size = word_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()
        self.img_encoder_type = img_encoder_type
        self.img_encoder = get_img_encoder( img_encoder_type, embed_size )

    def forward(self, img, qst):
        raise NotImplementedError()
    
    def generate(self, img):
        '''
        Generate question-answer pairs for the given
        images
        '''
        raise NotImplementedError()
    
    def genotype( self ):
        return self.img_encoder.darts.genotype()

    def arch_parameters( self ):
        return self.img_encoder.darts.arch_parameters()
    
    def save_arch_parameters( self, save_path ):
        if self.img_encoder_type != 'darts':
            return
        self.img_encoder.darts.save_arch_parameters( save_path )
    
    def load_arch_parameters( self, load_path ):
        if self.img_encoder_type != 'darts':
            return
        self.img_encoder.darts.load_arch_parameters( load_path )

class VqaModel(VqaModelBase):
    '''
    Original ( baseline ) VqaModel with question and answer generated
    seperately
    '''

    def __init__(self, embed_size, qst_vocab_size,
        ans_vocab_size, word_embed_size, num_layers,
        hidden_size, img_encoder_type='vgg'):

        super(VqaModel, self).__init__( embed_size, qst_vocab_size,
                ans_vocab_size, word_embed_size, num_layers, hidden_size,
                img_encoder_type )
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size,
                embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        # import pdb; pdb.set_trace()
        # [batch_size, embed_size]
        img_feature = self.img_encoder(img)
        # qst_feat: [batch_size, embed_size] qst_out: [bs, qst_vocab_size]
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
    
    def new(self):
        new_darts = self.img_encoder.darts.new()
        new_model = VqaModel(self.embed_size, self.qst_vocab_size,
                self.ans_vocab_size, self.word_embed_size, self.num_layers,
                self.hidden_size, self.img_encoder_type)
        new_model.img_encoder.darts = new_darts
        new_model.to( config.DEVICE )
        return new_model
   
    def _loss(self, images, questions, labels, qst_only=False):
        ans_out, qst_out = self( images, questions )
        # ans loss
        ans_loss = self.criterion( ans_out, labels )
        # question loss
        qst = questions[:, 1:].flatten()
        qst_out = qst_out[:, :-1].flatten(end_dim=1)
        qst_loss = self.criterion(qst_out, qst)
        # total loss
        if qst_only:
            loss = qst_loss
        else:
            loss = ans_loss + qst_loss
        return loss

class VqaModelUnified( VqaModelBase ):
    '''
    VqaModel that generates questiona and answer together
    '''
 
    def __init__(self, embed_size, unified_vocab_size,
        word_embed_size, num_layers, hidden_size, 
        img_encoder_type='vgg'):

        super(VqaModelUnified, self).__init__( embed_size, 
                unified_vocab_size, unified_vocab_size, 
                word_embed_size, num_layers, hidden_size,
                img_encoder_type )
        self.unified_vocab_size = unified_vocab_size
        self.qa_encoder = QstAnsEncoder(unified_vocab_size,
                word_embed_size, embed_size, num_layers, hidden_size)

    def forward( self, img, qa_str ):
        '''
        img: images
        qa_str: tokens representing question+answer
        returns a vector of shape bs x max_length x unified_vocab_size
        '''
        # import pdb; pdb.set_trace()
        # batch_size x embed_size
        img_feature = self.img_encoder(img)
        # bs x max_length x unified_vocab_size
        qa_out = self.qa_encoder(qa_str, img_feature)
        return qa_out
    
    def generate(self, img):
        '''
        Generate question-answer pairs for the given
        images
        '''
        # import pdb; pdb.set_trace()
        img_feature = self.img_encoder(img)
        # generate question+answer string
        qa_out = self.qa_encoder.generate(img_feature)
        return qa_out
    
    def new(self):
        new_darts = self.img_encoder.darts.new()
        new_model = VqaModelUnified(self.embed_size,
                self.unified_vocab_size, self.word_embed_size,
                self.num_layers, self.hidden_size,
                self.img_encoder_type)
        new_model.img_encoder.darts = new_darts
        new_model.to( config.DEVICE )
        return new_model
    
    def _loss(self, images, qa_gt, labels=None, qst_only=False):
        qa_out = self( images, qa_gt )
        qa_gt_flat = qa_gt[:, 1:].flatten()
        qa_out_flat = qa_out[:, :-1].flatten(end_dim=1)
        loss = self.criterion(qa_out_flat, qa_gt_flat)
        return loss

def test( model ):
    import pdb; pdb.set_trace()
    qst_vocab_size = model.qst_vocab_size
    ans_vocab_size = model.ans_vocab_size
    temperature = 1
    batch_size = 4
    img_size = 64
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss()
    img = torch.randn( batch_size, 3, img_size, img_size )
    qst = torch.randint( qst_vocab_size, ( batch_size, qst_max_len) )
    # test forward pass
    ans_out, qst_out = model( img, qst )
    assert ans_out.shape == ( batch_size, ans_vocab_size )
    assert qst_out.shape == ( batch_size, qst_max_len, qst_vocab_size )
    # test generate
    qst_gen, ans_gen = model.generate(img)
    assert qst_gen.shape == ( batch_size, qst_max_len )
    assert ans_gen.shape == ( batch_size, ans_vocab_size )
    if model.img_encoder_type == 'vgg':
        return
    # test architecture loss
    # loss = model.img_encoder.darts._loss( img, qst, labels )
    labels = torch.randint( ans_vocab_size, (batch_size, ) )
    new_model = model.new()
    loss = new_model._loss( img, qst, labels )
    print( 'Test passed!' )

def test_vgg():
    # global config.DEVICE
    config.DEVICE = 'cpu'
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 10
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    img_encoder_type = 'vgg'
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size, img_encoder_type )
    test( model )

def test_darts():
    # global config.DEVICE
    config.DEVICE = 'cpu'
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 10
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    img_encoder_type = 'darts'
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size, img_encoder_type )
    test( model )

def test_unified():
    # global config.DEVICE
    config.DEVICE = 'cpu'
    embed_size = 512
    unified_vocab_size = 8192
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    img_encoder_type = 'darts'
    model = VqaModelUnified( embed_size,
            unified_vocab_size, word_embed_size,
            num_layers, hidden_size, img_encoder_type )
    batch_size = 4
    img_size = 64
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss()
    img = torch.randn( batch_size, 3, img_size, img_size )
    qst = torch.randint( unified_vocab_size, ( batch_size, qst_max_len) )
    # test forward pass
    qa_out = model( img, qst )
    assert qa_out.shape == ( batch_size, qst_max_len, unified_vocab_size )
    qa_gen = model.generate( img )
    assert qa_gen.shape == ( batch_size, qst_max_len )
    # test architecture loss api
    new_model = model.new()
    loss = new_model._loss( img, qst )
    print( 'Test passed!' )

def main():
    test_vgg()
    test_darts()
    test_unified()

if __name__ == '__main__':
    main()
