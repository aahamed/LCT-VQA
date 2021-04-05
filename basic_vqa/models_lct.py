import torch
import torch.nn as nn
import torchvision.models as models
from pcdarts.model_search import Network

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
        pass
        # import pdb; pdb.set_trace()
        # image has dimensions [ batch_size, 3, 224, 224 ]
        img_feature = self.darts( image )
        # img_feature has dimensions [ batch_size, embed_size ]
        l2_norm = img_feature.norm(p=2, dim=1, keepdim=True).detach()
        # l2 normalized feature vector
        img_feature = img_feature.div(l2_norm)

        return img_feature


class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states

    def forward(self, question):

        # import pdb; pdb.set_trace()
        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=30, word_embed_size=300]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=30, batch_size, word_embed_size=300]
        _, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=2, batch_size, hidden_size=512]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=2, batch_size, 2*hidden_size=1024]
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=2, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=2048]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature


class VqaModel(nn.Module):

    def __init__(self, embed_size, qst_vocab_size, ans_vocab_size, word_embed_size, num_layers, hidden_size):

        super(VqaModel, self).__init__()
        self.img_encoder = ImgEncoder(embed_size, self)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, img, qst):

        # import pdb; pdb.set_trace()
        img_feature = self.img_encoder(img)                     # [batch_size, embed_size]
        qst_feature = self.qst_encoder(qst)                     # [batch_size, embed_size]
        combined_feature = torch.mul(img_feature, qst_feature)  # [batch_size, embed_size]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc1(combined_feature)           # [batch_size, ans_vocab_size=1000]
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.fc2(combined_feature)           # [batch_size, ans_vocab_size=1000]

        return combined_feature

def test_vqa():
    print( 'Test VQA model' )
    embed_size = 1024
    qst_vocab_size = 8192
    ans_vocab_size = 1000
    word_embed_size = 300
    num_layers = 1
    hidden_size = 512
    # import pdb; pdb.set_trace()
    model = VqaModel( embed_size, qst_vocab_size,
            ans_vocab_size, word_embed_size,
            num_layers, hidden_size )
    batch_size = 4
    img_size = 224
    qst_max_len = 30
    img = torch.randn( batch_size, 3, img_size, img_size )
    qst = torch.randint( qst_vocab_size, ( batch_size, qst_max_len) )
    out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    labels = torch.randint( ans_vocab_size, (batch_size, ) )
    loss = model.img_encoder.darts._loss( img, qst, labels )
    print( 'Test passed!' )

def test():
    test_vqa()

def main():
    test()

if __name__ == '__main__':
    main()
