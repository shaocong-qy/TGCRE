import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.metrics import precision_score, recall_score, f1_score


def compute_macro_PRF(predicted_idx, gold_idx, i=-1, empty_label=None):
    '''
    This evaluation function follows work from Sorokin and Gurevych(https://www.aclweb.org/anthology/D17-1188.pdf)
    code borrowed from the following link:
    https://github.com/UKPLab/emnlp2017-relation-extraction/blob/master/relation_extraction/evaluation/metrics.py
    '''
    if i == -1:
        i = len(predicted_idx)

    complete_rel_set = set(gold_idx) - {empty_label}
    avg_prec = 0.0
    avg_rec = 0.0

    for r in complete_rel_set:
        r_indices = (predicted_idx[:i] == r)
        tp = len((predicted_idx[:i][r_indices] == gold_idx[:i][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        avg_prec += prec
        avg_rec += rec
    f1 = 0
    avg_prec = avg_prec / len(set(predicted_idx[:i]))
    avg_rec = avg_rec / len(complete_rel_set)
    if (avg_rec + avg_prec) > 0:
        f1 = 2.0 * avg_prec * avg_rec / (avg_prec + avg_rec)

    return avg_prec, avg_rec, f1


def extract_relation_emb(model, testloader, mode):
    out_sentence_embs = None
    out_labels = None
    e1_hs = None
    e2_hs = None
    model.eval()
    for data in testloader:
        tokens_tensors, marked_e1, marked_e2, marked_head, marked_tail, \
        attention_mask, relation_emb, relation_head_emb, relation_tail_emb = [t.cuda() for t in data if t is not None]

        with torch.no_grad():
            outputs, out_sentence_emb, e1_h, e2_h, p1, p2 = model(input_ids=tokens_tensors,
                                                          e1_mask=marked_e1,
                                                          e2_mask=marked_e2,
                                                          marked_head=marked_head,
                                                          marked_tail=marked_tail,
                                                          attention_mask=attention_mask,
                                                          input_relation_emb=relation_emb,
                                                          input_relation_head_emb=relation_head_emb,
                                                          input_relation_tail_emb=relation_tail_emb
                                                          )

        if out_sentence_embs is None:
            out_sentence_embs = out_sentence_emb
            e1_hs = e1_h
            e2_hs = e2_h
        else:
            out_sentence_embs = torch.cat((out_sentence_embs, out_sentence_emb))
            e1_hs = torch.cat((e1_hs, e1_h))
            e2_hs = torch.cat((e2_hs, e2_h))

    return out_sentence_embs, e1_hs, e2_hs, p1 ,p2


# Return the performance of each relation
def single_relation_evaluate(args, predicted_idx, gold_idx, idx2relation=None):
    relation_num = args.unseen
    assert relation_num == len(set(gold_idx))
    length = len(predicted_idx)
    complete_rel_set = set(gold_idx)
    single_evaluate = {}
    error = {}
    target_rel = [] # Identify mis-classifications in specific relation, for example : target_rel = ['P123','P456']
    for r in complete_rel_set:
        r_indices = (predicted_idx[:length] == r)
        tp = len((predicted_idx[:length][r_indices] == gold_idx[:length][r_indices]).nonzero()[0])
        tp_fp = len(r_indices.nonzero()[0])
        tp_fn = len((gold_idx == r).nonzero()[0])
        prec = (tp / tp_fp) if tp_fp > 0 else 0
        rec = tp / tp_fn
        f1 = 0.0
        # Identify mis-classifications in specific relation
        if idx2relation[r] in target_rel:
            error_dict = {rel: 0 for idx, rel in idx2relation.items() if idx != r}
            target_indices = (gold_idx[:length] == r)
            l = len(target_indices.nonzero()[0])
            for i in predicted_idx[:length][target_indices]:
                if i != r:
                    error_dict[idx2relation[i]] += 1
            for i in error_dict:
                error_dict[i] /= (0.01 * l)
            error_tup = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)
            # top3 in mis-classifications
            error[idx2relation[r]] = error_tup[:3]
        if (rec + prec) > 0:
            f1 = 2.0 * prec * rec / (prec + rec)
        if idx2relation is not None:
            single_evaluate[idx2relation[r]] = {'f': round(f1, 4), 'p': round(prec, 4), 'r': round(rec, 4)}
        else:
            single_evaluate[r] = {'f': round(f1, 4), 'p': round(prec, 4), 'r': round(rec, 4)}
    if target_rel:
        print('target_relation_error: ')
        print(error)
    return single_evaluate


def evaluate(args, preds, e1_hs, e2_hs, y_attr, y_e1, y_e2, true_label, p1, p2, idx2relation=None):
    p1 = p1.detach().cpu()
    p2 = p2.detach().cpu()

    Mat_A = torch.stack((preds,e1_hs,e2_hs)).permute(1,0,2) # (3,relation_nums,768) ==> (relation_nums,3,768)
    Mat_B= torch.matmul(p1,Mat_A)
    Mat_C = torch.matmul(Mat_A,p2)
    # Mat_B = torch.matmul(mid,p2)

    Mat_AA = Mat_A.permute(1,0,2)   # (relation_nums,3,768) ==> (3,relation_nums,768)
    Mat_BB = Mat_B.permute(1,0,2)
    Mat_CC = Mat_C.permute(1, 0, 2)


    A_preds = Mat_AA[0].unsqueeze(0)
    A_e1_hs = Mat_AA[1].unsqueeze(0)
    A_e2_hs = Mat_AA[2].unsqueeze(0)

    B_preds = Mat_BB[0].unsqueeze(0)
    B_e1_hs = Mat_BB[1].unsqueeze(0)
    B_e2_hs = Mat_BB[2].unsqueeze(0)

    C_preds = Mat_CC[0].unsqueeze(0)
    C_e1_hs = Mat_CC[1].unsqueeze(0)
    C_e2_hs = Mat_CC[2].unsqueeze(0)




    y_attr = torch.tensor(y_attr)
    y_e1 = torch.tensor(y_e1)
    y_e2 = torch.tensor(y_e2)
    A = torch.stack((y_attr,y_e1,y_e2)).permute(1,0,2) # (3,relation_nums,768) ==> (relation_nums,3,768)
    B = torch.matmul(p1,A)
    C = torch.matmul(A,p2)
    # B = torch.matmul(mid_mat,p2)

    AA = A.permute(1,0,2)   # (relation_nums,3,768) ==> (3,relation_nums,768)
    BB = B.permute(1,0,2)   # (relation_nums,3,768) ==> (3,relation_nums,768)
    CC = C.permute(1,0,2)

    A_attr = AA[0].unsqueeze(1)
    A_e1 = AA[1].unsqueeze(1)
    A_e2 = AA[2].unsqueeze(1)

    B_attr = BB[0].unsqueeze(1)
    B_e1 = BB[1].unsqueeze(1)
    B_e2 = BB[2].unsqueeze(1)

    C_attr = CC[0].unsqueeze(1)
    C_e1 = CC[1].unsqueeze(1)
    C_e2 = CC[2].unsqueeze(1)



    A_sent_dist = torch.cosine_similarity(A_attr, A_preds, dim=-1)
    A_e1_dist = torch.cosine_similarity(A_e1, A_e1_hs, dim=-1)
    A_e2_dist = torch.cosine_similarity(A_e2, A_e2_hs, dim=-1)

    B_sent_dist = torch.cosine_similarity(B_attr, B_preds, dim=-1)
    B_e1_dist = torch.cosine_similarity(B_e1, B_e1_hs, dim=-1)
    B_e2_dist = torch.cosine_similarity(B_e2, B_e2_hs, dim=-1)

    C_sent_dist = torch.cosine_similarity(C_attr, C_preds, dim=-1)
    C_e1_dist = torch.cosine_similarity(C_e1, C_e1_hs, dim=-1)
    C_e2_dist = torch.cosine_similarity(C_e2, C_e2_hs, dim=-1)



    A_result = (1 - 2 * args.alpha) * A_sent_dist + args.alpha * A_e1_dist + args.alpha * A_e2_dist
    B_result = (1 - 2 * args.alpha) * B_sent_dist + args.alpha * B_e1_dist + args.alpha * B_e2_dist
    C_result = (1 - 2 * args.alpha) * C_sent_dist + args.alpha * C_e1_dist + args.alpha * C_e2_dist

    result = (A_result + B_result + C_result).numpy()
    predictions = result.argmax(axis=0)


    # sent_dist = cosine_similarity(y_attr, preds)
    # e1_dist = cosine_similarity(y_e1, e1_hs)
    # e2_dist = cosine_similarity(y_e2, e2_hs)

    # result = (1 - 2 * args.alpha) * sent_dist + args.alpha * e1_dist + args.alpha * e2_dist
    # predictions = result.argmax(axis=0)
    if idx2relation is not None:
        single_evaluate = single_relation_evaluate(args, predictions, np.array(true_label), idx2relation)
        sort_result = dict(sorted(single_evaluate.items(), key=lambda x: x[1]['f']))
        print('single_relation result: ')
        print(sort_result)
    p_macro, r_macro, f_macro = compute_macro_PRF(predictions, np.array(true_label))
    return p_macro, r_macro, f_macro
