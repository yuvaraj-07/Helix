from static.cannyeval_v3 import *
evaluatorA = CannyEval() 
def calc(num_q,teach_ans,student_ans):
    dj=dict()
    #{"1":["Sunrise and Sunset are good to see but we never can't expect them to see at the same time", [["Both Sunrise and Sunset are good to see"]]],"2":["all the animals in the world will go under hibernation in winter season", [["Animals will undergo hibernation in summer season"]]]}
    for i in range(num_q):
        dj[str(i+1)]=[str(teach_ans[i]),[[str(student_ans[i])]]]
    print(dj)
    #dj = json.load(open('./minimal-data-4.json'))
    report = evaluatorA.report_card(data_json=dj, max_marks=10, relative_marking=False, integer_marking=False, json_load_version="v2")
    print("Report of the evaluations")
    print(report)

    print("Information about student answers")
    print("Orienting sentences :", evaluatorA.orienting_sens), 
    print("Orienting phrases :", evaluatorA.orienting_phrases),
    print("Disorienting sentences :", evaluatorA.disorienting_sens)
    print("Disorienting phrases :", evaluatorA.disorienting_phrases)
num_q=2
teach_ans=["Sunrise and Sunset are good to see but we never can't expect them to see at the same time","all the animals in the world will go under hibernation in winter season"]
student_ans=[["Both Sunrise and Sunset are good to see"],"Animals will undergo hibernation in summer season"]
calc(num_q,teach_ans,student_ans)