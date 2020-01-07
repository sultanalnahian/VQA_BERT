import json
import pickle

class VQA:
    def __init__(self, vqa_file = None, answer_to_id_file = None, semantic_info_file = None):
        self.images = []
        self.questions = []
        self.answers = []
        self.ans_to_ix = dict()
        self.ix_to_ans = dict()
        self.img_to_semantic_info = dict()
        if vqa_file != None:
            vqa_dataset = json.load(open(vqa_file, 'r'))
            self.LoadAnnotations(vqa_dataset)
        if answer_to_id_file != None:
            answer_to_id_data = json.load(open(answer_to_id_file, 'r'))
            self.LoadAnswerToId(answer_to_id_data)
        if semantic_info_file != None:
            with open(semantic_info_file, 'rb') as f:
                semantic_info = pickle.load(f)
                self.loadSemanticInfo(semantic_info)

    def LoadAnnotations(self, vqa_dataset = None):
        images = []
        questions = []
        answers = []
        
        imdir='COCO_%s2014_%012d.jpg'
        for each_data in vqa_dataset:
            #question = each_data['final_question']
            question = each_data['question']
            ans = each_data['ans']
            image = each_data['img_id']
            image_path = imdir%('train',image)
            questions.append(question)
            answers.append(ans)
            images.append(image_path)

        self.images = images
        self.questions = questions
        self.answers = answers
    
    def LoadAnswerToId(self, answer_to_id_data):
        for ix, ans in answer_to_id_data.items():
            self.ans_to_ix[ans] = ix
            self.ix_to_ans[ix] = ans

    def loadSemanticInfo(self, image_to_description):
        for key, value in image_to_description.items():
            desc_str = ""
            for each_desc in value:
                desc_str = desc_str + " " + each_desc +"."
                desc_str = desc_str.strip()
            
            self.img_to_semantic_info[key] = desc_str
    
#vqa = VQA('./preprocessed_data/data_prepro_train_1000_ans.json', './preprocessed_data/ix_to_ans_1000.json')