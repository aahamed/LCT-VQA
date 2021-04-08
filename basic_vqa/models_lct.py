import torch
import torch.nn as nn
import torchvision.models as models
from pcdarts.model_search import Network
from constants import *

class ImgEncoder(nn.Module):

    def __init__(self, embed_size, vqa_model, init_ch=16, layers=4 ):
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

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoderFixed, self).__init__()
        model = models.vgg19(pretrained=True)
        in_features = model.classifier[-1].in_features  # input size of feature vector
        model.classifier = nn.Sequential(
            *list(model.classifier.children())[:-1])    # remove last fc layer

        self.model = model                              # loaded model without last fc layer
        self.fc = nn.Linear(in_features, embed_size)    # feature vector of image

    def forward(self, image):
        """Extract feature vector from image vector.
        """
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            img_feature = self.model(image)                  # [batch_size, vgg16(19)_fc=4096]
        img_feature = self.fc(img_feature)                   # [batch_size, embed_size]

        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        img_feature = img_feature.div(l2_norm)               # l2-normalized feature vector

        return img_feature


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

        # create start tokens
        start_word = torch.ones( (batch_size, 1) ).long().to( DEVICE )
        start_vec = self.word2vec( start_word )
        start_vec = self.tanh( start_vec )
        start_vec = start_vec.transpose(0, 1)

        # generate question
        hidden_state = ( hidden_state, hidden_state )
        current_word = start_vec
        qst_out = torch.zeros( (batch_size, self.max_length) )\
                .long().to( DEVICE )
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
            pred = torch.tensor( pred ).to( DEVICE )
        return pred



class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size, num_layers,
            hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = None
        if ARCH_TYPE == 'darts':
            self.img_encoder = ImgEncoder(embed_size, self)
        else:
            self.img_encoder = ImgEncoderFixed(embed_size)
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
        with torch.no_grad():
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

def test_vqa():
    global DEVICE
    DEVICE = 'cpu'
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
            num_layers, hidden_size )
    batch_size = 4
    img_size = 64
    qst_max_len = 30
    criterion = nn.CrossEntropyLoss()
    img = torch.randn( batch_size, 3, img_size, img_size )
    qst = torch.randint( qst_vocab_size, ( batch_size, qst_max_len) )
    # test forward pass
    out, qst_out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    assert qst_out.shape == ( batch_size, qst_max_len, qst_vocab_size )
    labels = torch.randint( ans_vocab_size, (batch_size, ) )
    # test architecture loss
    # loss = model.img_encoder.darts._loss( img, qst, labels )
    # test teacher forcing
    N = batch_size * qst_max_len
    qst = qst.view( N )
    qst_out = qst_out.view( N, -1 )
    qst_loss = criterion( qst_out[:-1], qst[1:] )
    # test qa generation
    qst, ans = model.generate( img )
    assert ans.shape == ( batch_size, ans_vocab_size )
    assert qst.shape == ( batch_size, qst_max_len )
    print( 'Test passed!' )

def test():
    test_vqa()

def main():
    test()

if __name__ == '__main__':
    main()