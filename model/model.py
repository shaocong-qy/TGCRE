import torch
import random
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
from transformers import AutoModel, BertModel, BertPreTrainedModel
from torch.autograd import Function
import math



def extract_entity(sequence_output, e_mask):
    extended_e_mask = e_mask.unsqueeze(-1)
    extended_e_mask = extended_e_mask.float() * sequence_output
    extended_e_mask, _ = extended_e_mask.max(dim=-2)
    # extended_e_mask = torch.stack([sequence_output[i,j,:] for i,j in enumerate(e_mask)])
    return extended_e_mask.float()


# class REMatchingModel(nn.Module):
class TGCRE(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.relation_emb_dim = config.hidden_size
        self.margin = torch.tensor(config.margin)
        self.alpha = config.alpha
        self.softmax=nn.Softmax(dim=-1)
        self.q_grad = nn.Parameter(torch.empty(1, config.maxlen))
        nn.init.xavier_normal_(self.q_grad)

        self.p1 = nn.Parameter(torch.randn(3, 3)) #,requires_grad=False
        nn.init.orthogonal( self.p1)
        self.p2 = nn.Parameter(torch.randn(768, 768))
        nn.init.orthogonal( self.p2)

        # self.pdist = nn.PairwiseDistance(p=2)

        #self.bert = AutoModel.from_pretrained(args.pretrained_model)
        self.bert = BertModel(config)
        self.fclayer = nn.Linear(self.relation_emb_dim , self.relation_emb_dim)
        # self.code = nn.Embedding(1, self.relation_emb_dim)
        self.classifier_grad = nn.Sequential(
                                        nn.Linear(self.relation_emb_dim, self.relation_emb_dim),
                                        nn.Tanh(),
                                        nn.Linear(self.relation_emb_dim, self.num_labels)
                                        )
        self.rev_loss = nn.CrossEntropyLoss()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            e1_mask=None,
            e2_mask=None,
            marked_head=None,
            marked_tail=None,
            input_relation_emb=None,
            input_relation_head_emb=None,
            input_relation_tail_emb=None,
            labels=None,
            num_neg_sample=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )

        sequence_output = outputs[0]  # Sequence of hidden-states of the last layer

        ###################################################################################################################
        # energy = torch.mean(torch.logsumexp(sequence_output, 1))
        # energy.backward()
        mloss = torch.tensor(0.).cuda()
        if labels is not None:
            token_embeds = torch.tensor(sequence_output, requires_grad=True)
            x = torch.mean(torch.logsumexp(token_embeds, 1))
            x.backward()
            # token_embedss=torch.logsumexp(token_embeds, 1)  #torch.logsumexp(sequence_output, 1)
            # token_embedss_logits = self.classifier_grad(token_embedss)
            # token_embedss_loss = self.rev_loss(token_embedss_logits.view(-1, self.num_labels), labels.view(-1))
            # token_embedss_loss.backward()

            grad = token_embeds.grad

            attribution = torch.abs(torch.sum(torch.mul(token_embeds, grad), dim=-1))
            # attribution = torch.sum(torch.mul(token_embeds, grad), dim=-1)
            attribution = attribution / torch.sum(attribution, dim=-1, keepdim=True)
            attribution = self.softmax(attribution)
            embeddings_grad = torch.matmul(attribution.unsqueeze(1), sequence_output).squeeze()

            q_mat = self.q_grad.repeat(sequence_output.size(0), 1, 1)
            embeddings_q = torch.matmul(q_mat, sequence_output).squeeze()

            scores = torch.cosine_similarity(embeddings_grad, embeddings_q)
            score = torch.sum(scores, dim=-1)/ sequence_output.size(0)



        else:

            q_mat = self.q_grad.repeat(sequence_output.size(0), 1, 1)
            embeddings_q = torch.matmul(q_mat, sequence_output).squeeze()

            # embeddings_q = self.fclayer(embeddings_q)

        # print(1)
        ###################################################################################################################


        e1_h = extract_entity(sequence_output, marked_head)
        e2_h = extract_entity(sequence_output, marked_tail)
        # e1_mark = extract_entity(sequence_output, e1_mask)
        # e2_mark = extract_entity(sequence_output, e2_mask)


        # pooled_output = torch.cat([e1_mark, e2_mark], dim=-1)
        # pooled_output = torch.cat([outputs[1], torch.mean(outputs[0],1)], dim=-1)
        # pooled_output = torch.cat([outputs[1], embeddings_q], dim=-1)



        # sentence_proj = emb_proj(sentence_embeddings, common_emb)
        # sentence_embeddings = emb_proj(sentence_embeddings, sentence_embeddings - sentence_proj)
        # sentence_embeddings = self.fclayer(embeddings_q)
        sentence_embeddings = torch.tanh(embeddings_q)
        e1_h = torch.tanh(e1_h)
        e2_h = torch.tanh(e2_h)





        outputs = (outputs,)
        if labels is not None:
            zeros = torch.tensor(0.).cuda()
            loss = torch.tensor(0.).cuda()


            # revloss = torch.tensor(0.).cuda()
            # rev_logit = self.classifier(embeddings_q)
            # revloss = self.rev_loss(rev_logit.view(-1, self.num_labels), labels.view(-1))
            mloss += torch.max(zeros,1-score)

            gamma = self.margin.cuda()

            for a, b in enumerate(zip(sentence_embeddings, e1_h, e2_h)):
                max_val = torch.tensor(0.).cuda()
                matched_sentence_pair = input_relation_emb[a]
                matched_head_pair = input_relation_head_emb[a]
                matched_tail_pair = input_relation_tail_emb[a]

                ###################################################################
                # print(torch.inverse(self.p1))
                # print(self.p1.transpose(0,1))
                Mat_A = torch.stack((b[0], b[1], b[2]))
                mid = torch.matmul(self.p1, Mat_A)
                Mat_B = torch.matmul(mid, self.p2)



                A = torch.stack((matched_sentence_pair,matched_head_pair,matched_tail_pair))
                mid_mat = torch.matmul(self.p1, A)
                B = torch.matmul(mid_mat, self.p2)

                B_sentence_pair = B[0]
                B_head_pair = B[1]
                B_tail_pair = B[2]

                ###################################################################

                # self.pdist
                #
                # pos_s = self.pdist(matched_sentence_pair, Mat_A[0]).cuda()
                # pos_h = self.pdist(matched_head_pair, Mat_A[1]).max().cuda()
                # pos_t = self.pdist(matched_tail_pair, Mat_A[2]).max().cuda()
                # pos_A = (1 - 2 * self.alpha) * pos_s + self.alpha * pos_h + self.alpha * pos_t
                #
                #
                # pos_s1 = self.pdist(B_sentence_pair, Mat_B[0], dim=-1).cuda()
                # pos_h1 = self.pdist(B_head_pair, Mat_B[1], dim=-1).max().cuda()
                # pos_t1 = self.pdisty(B_tail_pair, Mat_B[2], dim=-1).max().cuda()
                # pos_B = (1 - 2 * self.alpha) * pos_s1 + self.alpha * pos_h1 + self.alpha * pos_t1

                ###################################################################



                pos_s = torch.cosine_similarity(matched_sentence_pair, Mat_A[0], dim=-1).cuda()
                pos_h = torch.cosine_similarity(matched_head_pair, Mat_A[1], dim=-1).max().cuda()
                pos_t = torch.cosine_similarity(matched_tail_pair, Mat_A[2], dim=-1).max().cuda()
                pos_A = (1 - 2 * self.alpha) * pos_s + self.alpha * pos_h + self.alpha * pos_t


                pos_s1 = torch.cosine_similarity(B_sentence_pair, Mat_B[0], dim=-1).cuda()
                pos_h1 = torch.cosine_similarity(B_head_pair, Mat_B[1], dim=-1).max().cuda()
                pos_t1 = torch.cosine_similarity(B_tail_pair, Mat_B[2], dim=-1).max().cuda()
                pos_B = (1 - 2 * self.alpha) * pos_s1 + self.alpha * pos_h1 + self.alpha * pos_t1

                pos = (pos_A + pos_B)/2
                ########################  修改    ###########################################
                # context_describe = input_relation_emb[a]
                # head_describe = input_relation_head_emb[a]
                # tail_describe = input_relation_tail_emb[a]
                # context_instance = b[0]
                # head_instance= b[1]
                # tail_instance = b[2]
                #
                # matched_instance = torch.cat((head_instance,tail_instance,context_instance))
                # matched_entity =  torch.cat((head_describe,tail_describe,context_instance))
                # matched_context = torch.cat((head_instance,tail_instance,context_describe))
                # matched_describe = torch.cat((head_describe, tail_describe, context_describe))
                #
                # ditance_full = torch.cosine_similarity(matched_instance, matched_describe, dim=-1).cuda()
                # ditance_e = torch.cosine_similarity(matched_instance, matched_entity, dim=-1).max().cuda()
                # ditance_c = torch.cosine_similarity(matched_instance, matched_context, dim=-1).max().cuda()
                #
                # distance = (1 - 2 * self.alpha) * ditance_full + self.alpha * ditance_e + self.alpha * ditance_c
                ###################################################################

                if len(input_relation_emb) < num_neg_sample:
                    num_neg_sample = 1
                # randomly sample relation_emb
                rand = random.sample(range(len(input_relation_emb)), num_neg_sample)
                neg_relation_emb = torch.stack([input_relation_emb[i] for i in rand])
                neg_relation_head_emb = torch.stack([input_relation_head_emb[i] for i in rand])
                neg_relation_tail_emb = torch.stack([input_relation_tail_emb[i] for i in rand])
                for i, j in enumerate(zip(neg_relation_emb, neg_relation_head_emb, neg_relation_tail_emb)):
                    ########################  修改    #################################################################
                    # neg_context_describe = j[0]
                    # neg_head_describe = j[1]
                    # neg_tail_describe = j[2]
                    #
                    # matched_neg_entity = torch.cat((neg_head_describe, neg_tail_describe, context_instance))
                    # matched_neg_context = torch.cat((head_instance, tail_instance, neg_context_describe))
                    # matched_neg_describe = torch.cat((neg_head_describe, neg_tail_describe, neg_context_describe))
                    #
                    # tmp_full = torch.cosine_similarity(matched_instance, matched_neg_describe, dim=-1).cuda()
                    # tmp_e = torch.cosine_similarity(matched_instance, matched_neg_entity, dim=-1).max().cuda()
                    # tmp_c = torch.cosine_similarity(matched_instance, matched_neg_context, dim=-1).max().cuda()
                    # tmp = (1 - 2 * self.alpha) * tmp_full + self.alpha * tmp_e + self.alpha * tmp_c
                    #
                    # if tmp > max_val:
                    #     if (matched_describe == matched_neg_describe).all():
                    #         continue
                    #     else:
                    #         max_val = tmp
                    ########################  修改    #################################################################
                    A_neg = torch.stack((j[0],j[1],j[2]))
                    mid_mat_neg = torch.matmul(self.p1, A_neg)
                    B_neg = torch.matmul(mid_mat_neg, self.p2)


                    tmp_s = torch.cosine_similarity(Mat_A[0], j[0], dim=-1).cuda()
                    tmp_h = torch.cosine_similarity(Mat_A[1], j[1], dim=-1).max().cuda()
                    tmp_t = torch.cosine_similarity(Mat_A[2], j[2], dim=-1).max().cuda()
                    tmp_A = (1 - 2 * self.alpha) * tmp_s + self.alpha * tmp_h + self.alpha * tmp_t

                    tmp_s1 = torch.cosine_similarity(Mat_B[0], B_neg[0], dim=-1).cuda()
                    tmp_h1 = torch.cosine_similarity(Mat_B[1], B_neg[1], dim=-1).max().cuda()
                    tmp_t1 = torch.cosine_similarity(Mat_B[2], B_neg[2], dim=-1).max().cuda()
                    tmp_B = (1 - 2 * self.alpha) * tmp_s1 + self.alpha * tmp_h1 + self.alpha * tmp_t1

                    tmp = (tmp_A + tmp_B)/2

                    if tmp > max_val:
                        if (matched_sentence_pair == j[0]).all():
                            continue
                        else:
                            max_val = tmp

                neg = max_val.cuda()
                loss += torch.max(zeros, neg - pos + gamma)  #  loss += torch.max(zeros, neg - pos + gamma)
            outputs = (loss, mloss)
            #outputs = (loss, )
        return outputs, sentence_embeddings, e1_h, e2_h, self.p1 , self.p2
