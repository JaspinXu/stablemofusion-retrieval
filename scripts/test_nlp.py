import os
import spacy
nlp = spacy.load("en_core_web_sm")
def extract_subject_and_action(sentence):
        doc = nlp(sentence)
        subject = None
        before_subject = ""
        after_subject = ""

        for i, token in enumerate(doc):
            # 提取主语
            if "subj" in token.dep_:
                subject = token.text
                
                # 提取主语前的内容
                before_subject = ' '.join([t.text for t in doc[:i]]).strip()
                # 提取主语后的内容
                after_subject = ' '.join([t.text for t in doc[i + 1:]]).strip()
                after_subject = after_subject.rstrip('.')  # 去除句尾的句号
                break  # 找到主语后退出循环

        return [str(before_subject + ' ' + subject).strip()], [str(after_subject).strip()]
    
print(extract_subject_and_action('individual appears to raise arms.'))
# the person quickly moves his left leg to the left and brings his right leg quickly to the left also.
# individual appears to raise arms.
# a person takes a quick side step to its left.
# individual appears to raise arms.
# person takes a large step to the left fast
# individual appears to raise arms.
combined_sentences = []
subject1, actions1 = extract_subject_and_action('the person quickly moves his left leg to the left and brings his right leg quickly to the left also.')
_, actions2 = extract_subject_and_action('individual appears to raise arms.')

if subject1 and len(subject1) > 0:
    subject1 = subject1[0]
else:
    subject1 = ""
if actions1 and len(actions1) > 0:
    action1 = actions1[0]
else:
    action1 = ""  

if actions2 and len(actions2) > 0:
    action2 = actions2[0]
else:
    action2 = "" 

combined_text = f"{subject1} {action1} and {111} {action2}."
combined_sentences.append(combined_text)
print(combined_sentences)