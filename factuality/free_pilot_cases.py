"""Controlled construction targets, not independently annotated clinical cases."""
from .facts import freeze_expected,span


def build_cases():
    rows=[]
    def claim(cap,source,prediction,typ,quote,label):
        plan=freeze_expected(source,'history of present illness','',[])
        rows.append({'id':f'case-{len(rows)+1:02d}','source':source,'prediction':prediction,'section':'history of present illness','reference':'','plan':plan,'gold':{'capability':cap,'error':label in ['contradicted','unsupported'],'label':label,'fact_type':typ,'target_span':span(prediction,quote)}})
    claim('age','Patient: I am 57 years old.','The patient is 67 years old.','age','67','contradicted')
    claim('age','Doctor: The patient is 73 years old.','The patient is 37 years old.','age','37','contradicted')
    claim('age','Patient: I am 24 years old. My father is 68.','The patient is 68 years old.','age','68','contradicted')
    claim('age','Patient: I am 57 years old.','The patient is 57 years old.','age','57','supported')
    claim('age','Patient: I am fifty-seven years old.','The patient is 57 years old.','age','57','supported')
    claim('age','Patient: My father is 67 years old.','Father is 67 years old.','quantity','67','supported')
    claim('gender','Doctor: The patient is female.','The patient is male.','sex','male','contradicted')
    claim('gender','Patient: I identify as a woman.','The patient identifies as a man.','gender','man','contradicted')
    claim('gender','Patient: My pronouns are she/her. I have a cough.','He has a cough.','pronoun','He','contradicted')
    claim('gender','Doctor: The patient is female.','The patient is female.','sex','female','supported')
    claim('gender','Patient: I identify as nonbinary.','The patient identifies as non-binary.','gender','non-binary','supported')
    claim('gender','Patient: My brother is ill. He has a cough.','The brother is ill. He has a cough.','pronoun','He','supported')
    claim('unsupported','Patient: I have a cough.','The patient is 62 years old.','age','62','unsupported')
    claim('unsupported','Patient: I have a cough.','The patient takes metformin daily.','medication','metformin','unsupported')
    claim('unsupported','Patient: I have a cough.','The patient identifies as a woman.','gender','woman','unsupported')
    claim('unsupported','Patient: I am 57 years old.','The patient is 57 years old.','age','57','supported')
    claim('unsupported','Doctor: You have diabetes.','Diabetes is documented.','diagnosis','Diabetes','supported')
    claim('unsupported','Doctor: The diagnosis is asthma.','Asthma was diagnosed.','diagnosis','Asthma','supported')
    def omission(source,section,pred,quote,text,typ,label,required=True,fact_section=None):
        candidate={'id':'f1','quote':quote,'text':text,'fact_type':typ,'required':required,'provenance':'synthetic_construction'}
        if fact_section:candidate['section']=fact_section
        plan=freeze_expected(source,section,text,[candidate])
        rows.append({'id':f'case-{len(rows)+1:02d}','source':source,'section':section,'reference':text,'prediction':pred,'plan':plan,'gold':{'capability':'omission','error':label=='omitted','label':label,'expected_fact_id':'f1','fact_type':typ}})
    omission('Patient: I am allergic to penicillin.','allergies','Allergy review completed.','allergic to penicillin','Penicillin allergy','other','omitted')
    omission('Patient: I take aspirin 5 mg daily.','medications','Takes aspirin.','5 mg daily','Aspirin dose is 5 mg daily','quantity','omitted')
    omission('Doctor: Return tomorrow for follow-up.','plan','Follow-up discussed.','Return tomorrow','Return tomorrow','date','omitted')
    omission('Patient: I am allergic to penicillin.','allergies','Allergic to penicillin.','allergic to penicillin','Penicillin allergy','other','present')
    omission('Patient: I am 57 years old and have a cough.','chief complaint','Cough.','57 years old','Age 57','age','excluded',required=False)
    omission('Patient: My father has diabetes. I have a cough.','chief complaint','Cough.','My father has diabetes','Paternal diabetes','diagnosis','excluded',fact_section='family history')
    return rows
