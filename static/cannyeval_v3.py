
from sentence_transformers import SentenceTransformer
import sklearn
import numpy as np
import warnings
import re
from textblob import TextBlob
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm
import pandas as pd
#import bar_chart_race as bcr 
import copy
from math import *
#import matplotlib.animation as ani
import matplotlib.pyplot as plt
import json
import spacy
import pickle
import dill
#import neuralcoref    #commenting for now
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from fuzzywuzzy import process
#nlp = spacy.load('en_core_web_sm')
#neuralcoref.add_to_pipe(nlp)

warnings.filterwarnings("ignore")


class CannyEval():
  
  ABBR_VOCAB = dict()
  try:
    BIG_STRING = open('./abbreviation_vocab.txt').read()
  except:
    print("Please provide the file for accessing intial abbreviaiton vocab dictionary")


  def __init__(self):
    #dynamic imports
    
    self.json_obj = dict()
    self.csv_obj = {'teacher_answers':None, 'student_answers' : None}
    self.report = None
    self.gdm = None
    self.orienting_phrases = dict()
    self.disorienting_phrases = dict()
    self.orienting_sens = dict()
    self.disorienting_sens = dict()
    self.class_strength = 0
    self.semantic_match_weight = 0.8
    self.phrase_match_weight = 0.2
    pass

  # Online Python compiler (interpreter) to run Python online.
# Write Python 3 code in this online editor and run it.


  @staticmethod
  def _check_input(inp):
    val = inp['1']
    for ans in val[1]:
        assert type(ans) == type([]), 'Incompatable Json Format'
        for an in ans:
            assert type(an) == type("string"), 'Incompatable Json Format'
    return 
  
  
    

  @staticmethod
  def prepare_abbreviation_vocab(cls, updating_dictionary=dict() ):
    
    ABBR_VOCAB_met=cls.ABBR_VOCAB
    initial_count  = len(ABBR_VOCAB_met)
    
    if initial_count == 0:
    
      pattern = '.*'
      acronym = '<td style=.*>(.*)</td>'
      abbr = '<td>(.*)</td>'

      body = re.findall(pattern, cls.BIG_STRING)
     
      flag = 0
      for ele1 in body:
        
        if re.match(acronym, ele1):
          acr = re.findall(acronym, ele1)[0]
          continue
        if re.match(abbr, ele1):
          abr = re.findall(abbr, ele1)[0]
          flag = 1
        if flag == 1:
          flag = 0
          ABBR_VOCAB_met[acr] = abr
    
    ABBR_VOCAB_met.update(updating_dictionary)
    
    print("Total contents added : ", len(ABBR_VOCAB_met)-initial_count)
    print("Total contents of ABBR_VOCAB : ", len(ABBR_VOCAB_met))
    cls.ABBR_VOCAB = ABBR_VOCAB_met
    print(ABBR_VOCAB_met)
    return

  @staticmethod
  def fuzz_iou(setA, setB):
    ioun = 0
    ioud = len(setB) + 1e-6
    orph = set()
    disorph = set()

    for astring in setA:
      strOptions = setB
      Ratios = process.extract(astring,strOptions, limit=len(setB))
      
      flag = 0
      for ratio in Ratios:
        if ratio[1] >= 65:
          ioun += 1
          orph.add(astring)
          flag = 1
          break
      if flag == 0:
        disorph.add(astring)
    
    return ioun/ioud, (orph, disorph)  

  def abbreviate_in_sen(self, sen, abbr_vocab, show_abbreviated=True):
    #sen = 'KL is a powerful way to automate things today'
    acronyms = re.findall('[A-Z][A-Z]+',sen)
    for acronym in acronyms:
      try:
        sen = re.sub(acronym, abbr_vocab[acronym],sen)
      except:
        new_acronym = acronym
        abbr_vocab[new_acronym] = new_acronym #"<undefined abbreviation>"
        print('New Abbreviation Vocab', new_acronym, 'detected')
        sen = re.sub(acronym, abbr_vocab[acronym],sen)
    
    if show_abbreviated:
          print(sen)
    return sen
      


  def csv_from_json_v1(self, teacher_answers_json, student_answers_json):
    """converts teacher_answers in array form and student_answer in records json form to csvs"""

    tea_answers = pd.DataFrame({str(k+1):[v] for k,v in enumerate(teacher_answers_json)})
    question_count = len(tea_answers)
    arrlike_parsed = json.loads(student_answers_json)
    stu_answers = dict()
    for i in range(1, question_count+1):
        stu_answers[str(i)] = []
    for record in arrlike_parsed:
      for rk, rv in record.items():
        stu_answers[rk] += [rv]
    stu_answers = pd.DataFrame(stu_answers)

    return tea_answers, stu_answers, question_count

  def csv_from_json_v2(self, data_json, teacher_student_json_path="deprecated"):
    """converts json of format {question_id:[ teacher_answer, [student_scores], [[student_answers]]]} to csvs"""

    #jf = open(teacher_student_json_path)
    teacher_student_json = data_json #json.load(jf)
    tea_answers = pd.DataFrame({str(k):[v[0]] for k,v in teacher_student_json.items() })
    question_count = len(tea_answers.columns)

    stu_answers = dict()
    for i in range(1, question_count+1):
      stu_answers[str(i)] = []
    
    class_strength = len(teacher_student_json['1'][1])
    self.class_strength = class_strength

    for k,v in teacher_student_json.items():
      stu_answers[str(k)] = np.array(v[1]).reshape(-1)
      
      stu_answers[str(k)] = np.concatenate([stu_answers[str(k)], np.full((class_strength -stu_answers[str(k)].shape[0], ), np.nan)])
      
    stu_answers = pd.DataFrame(stu_answers)

    return tea_answers, stu_answers, question_count

  def ground_truth_scores(self, teacher_student_json_path):

    jf = open(teacher_student_json_path)
    teacher_student_json = json.load(jf)
    ground_truth_marks = pd.DataFrame({str(k):v[1] +[0 for i in range(self.class_strength-len(v[1]))] for k,v in teacher_student_json.items() })
    self.gdm = ground_truth_marks
    return ground_truth_marks

  def mae(self, ground_truth_marks, predicted_marks):
    gtm, pm = np.array(ground_truth_marks).astype(np.float64), np.array(predicted_marks)
    return np.mean(np.absolute(gtm - pm), axis = 0)

  
  def chunkize_abbreviate_correct(self, ans, abbr_vocab=None, spell_check=None):
    ans = ans.values
   
    answer_chunks_ref = []
    for index, an in enumerate(ans): 
      answer_chunk = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', an)
    

      answer_chunk_refined = []
    

      for se in answer_chunk:
        if len(se.split(" ")) >=250 :
          for part in range(len(se.split())//250):
            answer_chunk_refined.append(se[part*250:(part+1)*250])
        else:
          answer_chunk_refined.append(se)
      
      answer_chunk_ref = [TextBlob(self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False)).correct().string if spell_check == 'textblob' else self.abbreviate_in_sen(se, abbr_vocab, show_abbreviated=False) for se in answer_chunk_refined]
    
      answer_chunks_ref.append(answer_chunk_ref)
    
    
    return answer_chunks_ref

  def evaluate(self, student_answer=None, teacher_answer=None, encoders = [], abbr_vocab=ABBR_VOCAB, gen_context_spread=50, spell_check='None', student_id=-1, question_id=-1, encode_teacher_answer=True):

    #Acronym Replacement helps pick right match( because the trained data are highly likely to have seen abbreviations than acronyms and thus they know the context well when they have seen it)
    #Removing the stop words significantly improves the preference ratio but could affect negation statements try and use later
    #Passive and Active vocies are calculated well
    
    global qpembed1
    global qpembed2
    global qpembed3

    if student_answer == None or teacher_answer == None or abbr_vocab == None:

      print("Running Evaluation On Default Inputs")
      abbr_vocab = {'AI':'artificial intelligence', 'ML':'machine learning', 'DL' : 'deep learning', 'DS' : 'data science'}
      student_answer = 'The prominence of AI today has grown lot Artificial intelligence receives, understands and manipulates the world artificial intelligence is used to automate things'
      teacher_answer = 'AI is an important field present day Artificial intelligence means to be able to receive, understand, reason and change the environment AI is used to automate things'

      print("Student Answer : ")
      print(student_answer)

      print("Teacher Answer : ")
      print(teacher_answer)
    
    #INPUT PROCESSING =========================================================================================================================
    student_answer = re.sub("\\n+|\t+", " " , student_answer.strip())
    teacher_answer = re.sub("\\n+|\t+", " ", teacher_answer.strip())
    
    student_answer = re.sub("\[[0-9]+\]|:", "" , student_answer)
    teacher_answer = re.sub("\[[0-9]+\]|:", "", teacher_answer)
    
    #student_answer = self.corref_res(student_answer)    #commenting for now
    #teacher_answer = self.corref_res(teacher_answer)    #commenting for now

    #extracting the noun, date phrases
    student_answer_imps = self.extract_ImPs(student_answer)         
    teacher_answer_imps = self.extract_ImPs(teacher_answer)

    
    #calculating ImPs-orientation-scores 
    temp_jaccard, (orph, disorph) = self.fuzz_iou(student_answer_imps, teacher_answer_imps)
    

    if (question_id == -1 or student_id == -1) or (student_id == 1):
      self.orienting_phrases[question_id] = []
      self.disorienting_phrases[question_id] = []
    
    self.orienting_phrases[question_id].append(orph)  #{'1' :[ set(hitler, bad, cruel),  ] }
    self.disorienting_phrases[question_id].append(disorph)


    teacher_answer_chunks_ref, student_answer_chunks_ref = tea_answers_for_eval[str(question_id)][0], stu_answers_for_eval[str(question_id)][student_id-1]  # self.chunkize_abbreviate_correct(teacher_answer, student_answer, abbr_vocab, spell_check):
    
  
    
    #MODEL EVALUATION  ==========================================================================================================

    #models = ['sentence-transformers/paraphrase-MiniLM-L6-v2','distilbert-base-nli-stsb-mean-tokens'] #'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    correctnesses = []
    phrase_match_weight = 0.2
    semantic_match_weight = 0.8
    orph_sens_m = []
    disorph_sens_m = []
    for model in tqdm(range(len(encoders)), desc='Evaluating Student' +str(student_id)+' against Question' + str(question_id)+' ...', ncols=100):
      encoder = encoders[model]
      
      if encode_teacher_answer:
        embed_teacher_answer = encoder.encode(teacher_answer_chunks_ref)
        if model == 0:
          qpembed1 = embed_teacher_answer
        else:
          qpembed2 = embed_teacher_answer
      else:
        if model == 0:
          embed_teacher_answer = qpembed1
        else:
          embed_teacher_answer = qpembed2

      embed_student_answer = encoder.encode(student_answer_chunks_ref)

      correctness, (orph_sens, disorph_sens) = self.answer_similarities(embed_teacher_answer, embed_student_answer, question_id, student_id) #sklearn.metrics.pairwise.cosine_similarity(embed_teacher_answer, embed_student_answer, dense_output=True)[0][0] #rows are teacher_sens, cols are student_sens
      correctnesses.append(correctness*self.semantic_match_weight + self.phrase_match_weight*temp_jaccard)
      orph_sens_m += orph_sens
      disorph_sens_m += disorph_sens

    print("\nDone")
    correctnesses.append(np.mean(np.array(correctnesses), axis=0))
    
    if (question_id == -1 and student_id == -1) or (student_id == 1):
      self.orienting_sens[question_id] = []
      self.disorienting_sens[question_id] = []
    
    orph_sens_m = set(orph_sens_m)
    disorph_sens_m = set(disorph_sens_m)

    disorph_sens_m = disorph_sens_m - orph_sens_m
    self.orienting_sens[question_id].append(list(orph_sens_m))  #{'1' :[ [hitler, bad, cruel]  ] }
    self.disorienting_sens[question_id].append(list(disorph_sens_m))

    return correctnesses, teacher_answer_chunks_ref, student_answer_chunks_ref

  #def
  
  def answer_similarities(self, a, b, qid, stuid):
    repeat_penalty = [0 for z in range(b.shape[0])]
    repeat_count = [0 for z in range(b.shape[0])]
    orph_sens = set()
    disorph_sens = set()

    for i in range(b.shape[0]):
      repeat_penalty[i] = 1
      repeat_count[i] = 0
    repeat_penalty = np.array(repeat_penalty)
    repeat_count = np.array(repeat_count)

    scores_matrix = sklearn.metrics.pairwise.cosine_similarity(a, b)
    final_score = 0
    for index in range(a.shape[0]):
      stu_answer_id = np.argmax(scores_matrix[index, :])
      score = np.max(scores_matrix[index, :])

      repeat_count[stu_answer_id] += 1
      final_score += (1/a.shape[0])*score
    
    try:
      for saindex in range(b.shape[0]):
        if np.max(scores_matrix[:, saindex]) <= 0.75:
          disorph_sens.add(self.json_obj[str(qid)][1][stuid-1][saindex])
        else:
          orph_sens.add(self.json_obj[str(qid)][1][stuid-1][saindex])
    except:
      pass

    return final_score, (list(orph_sens), list(disorph_sens))
  
  def corref_res(self, text):
    
    doc = nlp(text)
    resolved_text = doc._.coref_resolved
    res = " ".join([sen.string.strip() for sen in nlp(resolved_text).sents])
    return res
  
  @staticmethod
  def get_longest_match(possibilities):
   
    res = []
    for pos in possibilities:
      
      maxi = 1
      tres = ''
      for match in pos:
        if len(match) > maxi:
            maxi = len(match)
            tres = match
     
      if tres != '':
        res += list(map(lambda x: x.lower(), tres.split()))
    
    return res

  def extract_ImPs(self, text):
    """Extracts ImPs like proper nouns, data, time objects etc which shouldn't be modified by student's 
    in their answeres and have to literally mention them as it is"""

    #Extracting propernouns
    tagged_sent = pos_tag(text.split())
    propernouns = [re.sub('\W', '', word.lower()) for word,pos in tagged_sent if pos == 'NNP' or pos == 'NNPS' or pos == "NN" or pos == "NNS"]

    
    #Extracting date formats : 1st January 2021, 01-01-2021, 01-01-21, January 1st, 2021
    date1 = '[0-9]{2}'
    date2 = '[0-9]?(1st|2nd|3rd|[4-9]th|0th)'
    month1 = 'January|February|March|April|May|June|July|August|September|October|November|December'
    month2 = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    month3 = '[0-9]{2}'
    year1 = '[0-9]{4}|[0-9]{2}'
    pattern1 = '(%s)?[-/\s,]?(%s)[-/\s,]?(%s)?'%(date2+'|'+ date1, month1+'|'+month2+'|'+month3, year1)
    pattern2 = '(%s)[-/\s,]?(%s)[-/\s,]?(%s)?'%(month1+'|'+ month2 + '|' + month3, date2+'|'+date1, year1)
    P = '.?(%s)|(%s)'%(pattern2, pattern1)
    try:
      dates = self.get_longest_match(re.findall(P, text))
    except:
      dates = []
    
    return set(dates+propernouns) 

  def report_card(self, data_json = None, teacher_student_answers_json=("Deprecated", None, None), teacher_student_answers_csv=(None,None), max_marks=5, relative_marking=False, integer_marking=False, json_load_version ="v2"):


    try:
      CannyEval._check_input(data_json)
      abbreviations = self.ABBR_VOCAB
      global tea_answers_for_eval, stu_answers_for_eval
      #DATAFRAME GENERATION ========================================================================================================
      if data_json != None:
        
        if json_load_version == "v1":
          tea_answers, stu_answers, question_count = self.csv_from_json_v1(teacher_student_answers_json[0], teacher_student_answers_json[1])   #Works only for JSON in format {ColumnName : {Index : RowValue}}   
        elif json_load_version == "v2":
          tea_answers, stu_answers, question_count = self.csv_from_json_v2(data_json, teacher_student_answers_json[0])

        tea_answers_for_eval = tea_answers.apply(self.chunkize_abbreviate_correct, abbr_vocab = abbreviations, spell_check='none')
        stu_answers_for_eval = stu_answers.apply(self.chunkize_abbreviate_correct, abbr_vocab = abbreviations, spell_check='none')

        for qid in range(question_count):
          self.json_obj[str(qid+1)] = [tea_answers_for_eval[str(qid+1)][0], stu_answers_for_eval[str(qid+1)].values.tolist()]
          
                    
      else:  
        stu_answers = pd.read_csv(teacher_student_answers_csv[1])
        tea_answers = pd.read_csv(teacher_student_answers_csv[0])
        question_count = len(tea_answers.columns)
        self.csv_obj['teacher_answers'], self.csv_obj['student_answers'] = tea_answers, stu_answers
      
      
      #OUTPUT DATAFRAME PREP ================================================================================================================
      class_strength = len(stu_answers)
      student_marks = dict()
      for q in range(question_count):
        student_marks["Question " + str(q+1)] = []

      #MODEL SELECTION =======================================================================================================================
      models = ['sentence-transformers/paraphrase-MiniLM-L6-v2','distilbert-base-nli-stsb-mean-tokens']
      encoders = []
      for model in models:
        encoder = SentenceTransformer(model)
        encoders.append(encoder)
      
      #EVALUATION CALLING ON ANSWER PAIRS ======================================================================================================
      encode_ta = True
      for col in range(1, question_count+1):
          teacher_answer = tea_answers[str(col)][0]
          encode_ta = True
          for stu in range(class_strength):
            if stu_answers[str(col)][stu] != str(np.nan): #len(teacher_answer.split(" "))
              correctnesses, _, _ = self.evaluate(stu_answers[str(col)][stu], teacher_answer,encoders=encoders, abbr_vocab=abbreviations, gen_context_spread=125, spell_check='None', student_id = stu+1, question_id=col, encode_teacher_answer=encode_ta) #smaller context spread is more specific but less accurate at encoding
              to_append = correctnesses[-1]
            else:
              to_append = 0
            
            if to_append <0.6:
              to_append = max(0, to_append - (1-to_append)/4) #(correctnesses[-1] -(1 - correctnesses[-1])/2) if correctnesses[-1] <0.6 else correctnesses[-1]) #human_induced bias = sqrt(distance_from_high_score)
            elif to_append >=0.95:
              to_append = 1.
            else:
              pass

            student_marks["Question " + str(col)].append(to_append)
            encode_ta = False
      
      #REPORT PREP ===============================================================================================================================
      report = student_marks
      report_norm = np.empty((question_count, class_strength)) #it's row = class_strength*questions = len
      for r in range(1, question_count+1):
          report_norm[r-1] = report["Question " + str(r)]

      report_norm_copy = copy.deepcopy(report_norm) #max mark
    
      for r in range(question_count):
        true_max =  max(report_norm_copy[r]) if relative_marking else 1
        true_min = 0 
        for c in range(class_strength):
          unrefined_score = (report_norm_copy[r][c]-true_min)/(true_max-true_min)*max_marks
          unrefined_score = np.round(unrefined_score, decimals=2)
          report_norm[r][c] = np.rint(unrefined_score) if integer_marking else  unrefined_score # mark = normalize(percentage), base_min_mark = 0, base_max_mark = true_max

      for q in range(question_count):
        student_marks["Question " + str(q+1)] = report_norm[q]
        
      report_csv = pd.DataFrame(student_marks)
      report_csv["Student Aggregate"] = np.sum(report_norm, axis=0)
      try:
        report_csv["Timestamp"] = stu_answers["Timestamp"]
        report_csv["Username"] = stu_answers["Username"]
        report_csv["Rollno"] = stu_answers["Rollno"]
      except:
        pass
      
      extra_info = dict()
      for k, v in enumerate(np.mean(report_norm, axis=1)):
        extra_info["Question " + str(k+1)] = v

      extra_info["Student Aggregate"] = np.mean(report_csv["Student Aggregate"])
      extra_info = pd.DataFrame(extra_info, index=["Class Mean"])
      report_csv = pd.concat([report_csv, extra_info], axis=0)

      
      return report_csv
    except AssertionError as msg:
      return msg

  def get_modstu_answers(self, ques_id=0, stu_id=0):
    teacher_student_json = self.json_obj
    return teacher_student_json[str(ques_id+1)][0], teacher_student_json[str(ques_id+1)][2][2+stu_id]
  

 
