import json
from tqdm import tqdm

def make_new_entries(annotations_info, questions_info):
    data = []
    for i in tqdm(range(len(annotations_info['annotations']))):
        ans = annotations_info['annotations'][i]['multiple_choice_answer']
        question_id = annotations_info['annotations'][i]['question_id']
        image_id = annotations_info['annotations'][i]['image_id']
        question = questions_info['questions'][i]['question']
        q_question_id = questions_info['questions'][i]['question_id']
        if question_id != q_question_id:
            print("question id does not match")
            raise

        data.append({'ques_id': question_id, 'question': question, 'ans': ans, 'img_id':image_id})
    return data

def preprocess_vqadata():
    train = []
    val = []
    train_anno = json.load(open('annotations/v2_mscoco_train2014_annotations.json', 'r'))
    val_anno = json.load(open('annotations/v2_mscoco_val2014_annotations.json', 'r'))

    train_ques = json.load(open('annotations/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))
    val_ques = json.load(open('annotations/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))
    
    train = make_new_entries(train_anno, train_ques)
    val = make_new_entries(val_anno, val_ques)
        
    print ('Training sample %d, Testing sample %d...' %(len(train), len(val)))

    json.dump(train, open('vqa_redefined_train.json', 'w'))
    json.dump(val, open('vqa_redefine_val.json', 'w'))

def main():
    preprocess_vqadata()

if __name__ == '__main__':
    main()