


from __future__ import print_function
import sys


def get_ner_fmeasure(golden_lists, predict_lists, label_type="BMES"):
    sent_num = len(golden_lists)
    golden_full = []
    predict_full = []
    right_full = []
    right_tag = 0
    all_tag = 0
    for idx in range(0,sent_num):
        # word_list = sentence_lists[idx]
        golden_list = golden_lists[idx]
        predict_list = predict_lists[idx][0:len(golden_list)]
        for idy in range(len(golden_list)):
            if golden_list[idy] == predict_list[idy]:
                right_tag += 1
        all_tag += len(golden_list)
        if label_type == "BMES" or label_type == "BIOES":
            gold_matrix = get_ner_BMES(golden_list)
            pred_matrix = get_ner_BMES(predict_list)
        else:
            gold_matrix = get_ner_BIO(golden_list)
            pred_matrix = get_ner_BIO(predict_list)
        # print "gold", gold_matrix
        # print "pred", pred_matrix
        right_ner = list(set(gold_matrix).intersection(set(pred_matrix)))
        golden_full += gold_matrix
        predict_full += pred_matrix
        right_full += right_ner
    right_num = len(right_full)
    golden_num = len(golden_full)
    predict_num = len(predict_full)
    if predict_num == 0:
        precision = -1
    else:
        precision =  (right_num+0.0)/predict_num
    if golden_num == 0:
        recall = -1
    else:
        recall = (right_num+0.0)/golden_num
    if (precision == -1) or (recall == -1) or (precision+recall) <= 0.:
        f_measure = -1
    else:
        f_measure = 2*precision*recall/(precision+recall)
    accuracy = (right_tag+0.0)/all_tag
    # print "Accuracy: ", right_tag,"/",all_tag,"=",accuracy
    if  label_type.upper().startswith("B-"):
        print("gold_num = ", golden_num, " pred_num = ", predict_num, " right_num = ", right_num)
    else:
        print("Right token = ", right_tag, " All token = ", all_tag, " acc = ", accuracy)
    return precision, recall, f_measure, accuracy


def reverse_style(input_string):
    print(input_string)
    target_position = input_string.index('[')
    input_len = len(input_string)
    output_string = input_string[target_position:input_len] + input_string[0:target_position]
    return output_string
    
    
def get_ner_demo(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B_'
    end_label = 'E_'
    single_label = 'S_'
    tag_index = []
    tag_type = ''
    tag_list = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        if label_list[i] != 0:
            current_label = label_list[i].upper()
        else:
            continue
        if begin_label in current_label:
            #实体开头
            if tag_type != '':
                tag_index.append((i-1))
                tag_index.append(tag_type)
                tag_list.append(tag_index)
                tag_index = []
                tag_type = ""
            tag_index.append(i)   #实体开始的位置
            tag_type = current_label.replace(begin_label,'',1) #实体的类型
            # print(tag_index,tag_type,tag_list)

        elif single_label in current_label:
            #单字实体
            if tag_type != '':
                tag_index.append((i-1))
                tag_index.append(tag_type)
                tag_list.append(tag_index)
                tag_index=[]
                tag_type = ""
            tag_index.append(i)
            tag_type = current_label.replace(single_label,'',1) #实体的类型
            tag_index.append(tag_type)
            tag_list.append(tag_index)
            # print(tag_index,tag_type,tag_list)
            tag_index = []
            tag_type = ""
            
        elif end_label in current_label:
            #实体结尾
            if tag_type != '':
                tag_index.append(i)
                tag_index.append(tag_type)
                tag_list.append(tag_index)                    
            tag_index = []
            tag_type = ''
            # print(tag_index,tag_type,tag_list)
        else:
            continue
    if (tag_index != [])&(tag_type != ''):
        tag_index.append(tag_type)
        tag_list.append(tag_index)
    ###原是列表，按要求转为字典        
    entity={}
    for l in tag_list:
        if l[-1] not in entity.keys():
            entity[l[-1]]=[l[0:2]]
        else:
            entity[l[-1]] += [l[0:2]]
    return tag_list,entity


def get_ner_BMES(label_list):
    # list_len = len(word_list)
    # assert(list_len == len(label_list)), "word list size unmatch with label list"
    list_len = len(label_list)
    begin_label = 'B_'
    end_label = 'E_'
    single_label = 'S_'
    whole_tag = ''
    index_tag = ''
    tag_list = []
    stand_matrix = []
    for i in range(0, list_len):
        # wordlabel = word_list[i]
        
        current_label = label_list[i].upper()
        # if 'DATE' in current_label:
 #            continue
        if begin_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(begin_label,"",1) +'[' +str(i)
            index_tag = current_label.replace(begin_label,"",1)

        elif single_label in current_label:
            if index_tag != '':
                tag_list.append(whole_tag + ',' + str(i-1))
            whole_tag = current_label.replace(single_label,"",1) +'[' +str(i)
            tag_list.append(whole_tag)
            whole_tag = ""
            index_tag = ""
        elif end_label in current_label:
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag+',' + str(i))
            whole_tag = ''
            index_tag = ''
        elif current_label == 'O':
            if (whole_tag != '')&(index_tag != ''):
                tag_list.append(whole_tag +',' + str(i-1))
            whole_tag = ''
            index_tag = ''

    if (whole_tag != '')&(index_tag != ''):
        tag_list.append(whole_tag)
    tag_list_len = len(tag_list)

    for i in range(0, tag_list_len):
        if  len(tag_list[i]) > 0:
            tag_list[i] = tag_list[i]+ ']'
            insert_list = reverse_style(tag_list[i])
            stand_matrix.append(insert_list)
    return stand_matrix





def readSentence(input_file):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    labels = []
    sentence = []
    label = []
    for line in in_lines:
        if len(line) < 2:
            sentences.append(sentence)
            labels.append(label)
            sentence = []
            label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            label.append(pair[-1])
    return sentences,labels


def readTwoLabelSentence(input_file, pred_col=-1):
    in_lines = open(input_file,'r').readlines()
    sentences = []
    predict_labels = []
    golden_labels = []
    sentence = []
    predict_label = []
    golden_label = []
    for line in in_lines:
        if "##score##" in line:
            continue
        if len(line) < 2:
            sentences.append(sentence)
            golden_labels.append(golden_label)
            predict_labels.append(predict_label)
            sentence = []
            golden_label = []
            predict_label = []
        else:
            pair = line.strip('\n').split(' ')
            sentence.append(pair[0])
            golden_label.append(pair[1])
            predict_label.append(pair[pred_col])

    return sentences,golden_labels,predict_labels


def fmeasure_from_file(golden_file, predict_file, label_type="BMES"):
    print("Get f measure from file:", golden_file, predict_file)
    print("Label format:",label_type)
    golden_sent,golden_labels = readSentence(golden_file)
    predict_sent,predict_labels = readSentence(predict_file)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%sm R:%s, F:%s"%(P,R,F))



def fmeasure_from_singlefile(twolabel_file, label_type="BMES", pred_col=-1):
    sent,golden_labels,predict_labels = readTwoLabelSentence(twolabel_file, pred_col)
    P,R,F = get_ner_fmeasure(golden_labels, predict_labels, label_type)
    print ("P:%s, R:%s, F:%s"%(P,R,F))



if __name__ == '__main__':
    # print "sys:",len(sys.argv)
    # if len(sys.argv) == 3:
#         fmeasure_from_singlefile(sys.argv[1],"BMES",int(sys.argv[2]))
#     else:
#         fmeasure_from_singlefile(sys.argv[1],"BMES")
    act = [['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'E_BOOK', 'O', 'S_LOC', 'B_PER', 'I_PER', 'E_PER'],
           ['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'E_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER'],
           ['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'E_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER']]
           
    # pre = [['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'I_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER'],
           # ['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'I_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER'],
           # ['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'I_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER']]
    
    # print(get_ner_fmeasure(act,pre))
    print(get_ner_demo(act))
    # print(get_ner_BMES(['B_TIME', 'I_TIME', 'I_TIME', 'I_TIME','E_TIME', 'O', 'B_BOOK', 'I_BOOK', 'O', 'O', 'B_PER', 'I_PER', 'E_PER']))