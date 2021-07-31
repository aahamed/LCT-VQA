import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import config

def softXEnt( pred, target ):
    '''Soft cross entropy loss'''
    logprobs = F.log_softmax( pred, dim=1 )
    return -( target * logprobs ).sum() / pred.shape[0]

class ImgEncoder(nn.Module):

    def __init__(self, embed_size):
        """(1) Load the pretrained model as you want.
               cf) one needs to check structure of model using 'print(model)'
                   to remove last fc layer from the model.
           (2) Replace final fc layer (score values from the ImageNet)
               with new fc layer (image feature).
           (3) Normalize feature vector.
        """
        super(ImgEncoder, self).__init__()
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
        self.img_encoder = ImgEncoder(embed_size)
        self.qst_encoder = QstEncoder(qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(embed_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

        self.embed_size = embed_size
        self.qst_vocab_size = qst_vocab_size
        self.ans_vocab_size = ans_vocab_size
        self.word_embed_size = word_embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.criterion = nn.CrossEntropyLoss()

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

    def new(self):
        new_model = VqaModel(self.embed_size, self.qst_vocab_size,
                self.ans_vocab_size, self.word_embed_size, self.num_layers,
                self.hidden_size)
        new_model.to( config.DEVICE )
        return new_model
   
    def _loss(self, images, questions, labels):
        ans_out = self( images, questions )
        loss = self.criterion( ans_out, labels )
        return loss

    def _soft_loss(self, images, questions, labels, pseudo_qst,
            pseudo_labels):
        ans_out_1 = self( images, questions )
        loss_1 = self.criterion( ans_out_1, labels )
        ans_out_2 = self( images, pseudo_qst )
        loss_2 = softXEnt( ans_out_2, pseudo_labels )
        loss = loss_1 + config.W_LAMBDA * loss_2
        return loss

def test():
    # global config.DEVICE
    config.DEVICE = 'cpu'
    temperature = config.TEMPERATURE
    embed_size = 512
    qst_vocab_size = 8192
    ans_vocab_size = 10
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
    out = model( img, qst )
    assert out.shape == ( batch_size, ans_vocab_size )
    # test architecture loss
    # loss = model.img_encoder.darts._loss( img, qst, labels )
    labels = torch.randint( ans_vocab_size, (batch_size, ) )
    soft_labels = torch.rand( batch_size, ans_vocab_size )
    soft_labels = F.softmax( soft_labels / temperature, dim=1 )
    new_model = model.new()
    loss = new_model._loss( img, qst, labels )
    soft_loss = new_model._soft_loss( img, qst, soft_labels )
    print( 'Test passed!' )

def main():
    test()

if __name__ == '__main__':
    main()
