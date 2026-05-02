
import joblib, time, os
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from transformers import pipeline as hf_pipeline
import warnings; warnings.filterwarnings('ignore')

app = FastAPI(title='NaloRH API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

print('NaloRH API - Chargement modeles...')
bundle      = joblib.load('/content/NaloRH_churn_model_v1.pkl')
churn_model = bundle['model']
FEATURES    = bundle['features']
THRESHOLD   = bundle['threshold']
nlp_pipe    = hf_pipeline('text-classification',
    model='nlptown/bert-base-multilingual-uncased-sentiment',
    truncation=True, max_length=512)
print(f'Churn model : {bundle["model_type"]} F1={bundle["f1_score"]:.3f}')
print('NLP model   : CamemBERT OK')
print('API prete')

class FeedbackInput(BaseModel):
    texte: str = Field(..., min_length=10)
    employe_id: Optional[str] = 'ANONYME'
    departement: Optional[str] = 'Non renseigne'
    monthly_income: Optional[float] = 5000
    years_at_company: Optional[float] = 3
    years_since_last_promotion: Optional[float] = 1
    overtime: Optional[bool] = False
    age: Optional[float] = 35
    job_satisfaction: Optional[float] = 2.5
    environment_satisfaction: Optional[float] = 2.5
    work_life_balance: Optional[float] = 2.5
    distance_from_home: Optional[float] = 10
    total_working_years: Optional[float] = 8
    job_level: Optional[float] = 2

class BatchInput(BaseModel):
    feedbacks: List[FeedbackInput]

class AnalyzeResponse(BaseModel):
    employe_id: str
    sentiment_label: str
    sentiment_score: float
    churn_risk_pct: float
    churn_risk_level: str
    themes: List[str]
    recommandation: str
    processing_ms: float

def get_sentiment(text):
    r = nlp_pipe(text)[0]
    stars = int(r['label'].split()[0]); conf = r['score']
    score = round((stars-1)/4*conf + 0.5*(1-conf), 3)
    label = 'Negatif' if stars<=2 else ('Neutre' if stars==3 else 'Positif')
    return {'label':label,'score':score}

def extract_themes(text):
    kws = {'Management':['manager','direction','responsable'],
           'Salaire':['salaire','remuneration','augmentation'],
           'Ambiance':['ambiance','equipe','collegues'],
           'Evolution':['evolution','promotion','carriere'],
           'Charge':['surcharge','heures','stress','burn-out'],
           'Formation':['formation','competences'],
           'Reconnaissance':['reconnaissance','valorise','apprecie']}
    txt = text.lower()
    return [t for t,ks in kws.items() if any(k in txt for k in ks)][:5] or ['General']

def build_features(inp, s):
    sal_exp = inp.monthly_income/(inp.total_working_years+1)
    stress  = (1 if inp.overtime else 0)*3 + (inp.distance_from_home/50)*2
    stag    = inp.years_since_last_promotion/(inp.years_at_company+1)
    sat_c   = s*0.4+inp.job_satisfaction/4*0.2+inp.environment_satisfaction/4*0.2+inp.work_life_balance/4*0.2
    m = {'sentiment_score':s,'satisfaction_composite':sat_c,
         'JobSatisfaction':inp.job_satisfaction,'EnvironmentSatisfaction':inp.environment_satisfaction,
         'WorkLifeBalance':inp.work_life_balance,'RelationshipSatisfaction':inp.job_satisfaction,
         'JobInvolvement':2.5,'MonthlyIncome':inp.monthly_income,'DailyRate':inp.monthly_income/22,
         'HourlyRate':inp.monthly_income/176,'PercentSalaryHike':12,'StockOptionLevel':0,
         'salary_exp_ratio':sal_exp,'JobLevel':inp.job_level,'TotalWorkingYears':inp.total_working_years,
         'YearsAtCompany':inp.years_at_company,'YearsInCurrentRole':inp.years_at_company*0.6,
         'YearsSinceLastPromotion':inp.years_since_last_promotion,
         'YearsWithCurrManager':inp.years_at_company*0.5,'NumCompaniesWorked':2,
         'TrainingTimesLastYear':2,'stagnation_ratio':stag,
         'loyalty_score':(inp.years_at_company*0.5)/(inp.years_at_company+1),
         'Age':inp.age,'Education':3,'PerformanceRating':3,
         'overtime_enc':1 if inp.overtime else 0,'travel_enc':1,
         'DistanceFromHome':inp.distance_from_home,'stress_score':stress,
         'dept_enc':0,'role_enc':0,'marital_enc':1,'gender_enc':0,'edfield_enc':0}
    return np.array([m.get(f,0.0) for f in FEATURES]).reshape(1,-1)

def reco(pct,themes):
    base = ('URGENT - Entretien immediat.' if pct>=70 else
            'Suivi renforce dans 2 semaines.' if pct>=45 else
            'Stable. Maintenir engagement.')
    tips = {'Management':' Revoir management.','Salaire':' Evaluer vs marche.',
            'Evolution':' Plan de carriere.','Charge':' Auditer charge.'}
    t = [tips[x] for x in themes if x in tips]
    return base + (t[0] if t else '')

@app.get('/health')
def health():
    return {'status':'ok','service':'NaloRH API','version':'1.0.0',
            'model_type':bundle['model_type'],'f1_score':bundle['f1_score']}

@app.get('/model/info')
def model_info():
    return {k:v for k,v in bundle.items() if k!='model'}

@app.post('/analyze', response_model=AnalyzeResponse)
def analyze(inp: FeedbackInput):
    t0=time.time()
    sent=get_sentiment(inp.texte); themes=extract_themes(inp.texte)
    prob=float(churn_model.predict_proba(build_features(inp,sent['score']))[0][1])
    pct=round(prob*100,1)
    lvl=('CRITIQUE' if prob>=THRESHOLD+0.10 else 'ELEVE' if prob>=THRESHOLD
         else 'MOYEN' if prob>=0.30 else 'FAIBLE')
    return AnalyzeResponse(employe_id=inp.employe_id,sentiment_label=sent['label'],
        sentiment_score=sent['score'],churn_risk_pct=pct,churn_risk_level=lvl,
        themes=themes,recommandation=reco(pct,themes),
        processing_ms=round((time.time()-t0)*1000,1))

@app.post('/analyze/batch')
def batch(data: BatchInput):
    t0=time.time(); res=[analyze(f) for f in data.feedbacks]
    high=sum(1 for r in res if r.churn_risk_level in ['ELEVE','CRITIQUE'])
    return {'total':len(res),'score_moyen':round(sum(r.sentiment_score for r in res)/len(res),3),
            'churn_eleves':high,'pct_a_risque':round(high/len(res)*100,1),
            'processing_ms':round((time.time()-t0)*1000,1),
            'resultats':[r.dict() for r in res]}
