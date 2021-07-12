import torch

def num_correct( pred, multi_choice ):
        res = torch.stack([(ans == pred) for ans in multi_choice])
        res = res.any(dim=0).sum().item()
        return res

def num_correct_qst( qst_pred, qst ):
    '''
    Number of correct questions
    '''
    qst_pred = qst_pred.argmax( dim=2 )[:, :-1]
    qst = qst[:, 1:]
    not_match = ~( qst == qst_pred )
    err_cnt = not_match.sum( dim=1 )
    acc_0 = ( err_cnt == 0 ).sum().item()
    acc_3 = ( err_cnt <= 3 ).sum().item()
    acc_5 = ( err_cnt <= 5 ).sum().item()
    return acc_0, acc_3, acc_5

