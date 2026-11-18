"""Retrieval-only experiments on the AI Citizen Assistant repo (no LLM). Usage: python3 -I run_experiments.py REPO_DIR"""
import sys, json, time, os, tempfile, statistics as st
REPO = sys.argv[1]; sys.path.insert(0, REPO); os.chdir(REPO)
import numpy as np
from sklearn.metrics import roc_auc_score
import core
HERE = os.path.dirname(os.path.abspath(__file__)) + "/"
emb = core.SentenceEmbedder()
try:
    emb._model(["probe"]) if emb.backend=="chroma_onnx" else emb._model.encode(["probe"]); ACTUAL=emb.backend
except Exception: ACTUAL="hash_vectorizer"   # repo silently falls back while still reporting its nominal backend
print("nominal backend:", emb.backend, "| actually used:", ACTUAL)

# ---------- E1: scheme retrieval, EN vs HI. (query, index of the correct scheme row in schemes.csv)
Q = [
 ("What documents do I need for a post matric scholarship?","पोस्ट मैट्रिक छात्रवृत्ति के लिए कौन से दस्तावेज़ चाहिए?",0),
 ("Which scholarship supports SC ST OBC students after class 10?","कक्षा 10 के बाद एससी एसटी ओबीसी छात्रों को कौन सी छात्रवृत्ति मिलती है?",0),
 ("Who is eligible for the farmer income support of 6000 rupees per year?","किसानों को सालाना 6000 रुपये की आय सहायता के लिए कौन पात्र है?",1),
 ("How do farmers register for PM Kisan Samman Nidhi?","किसान पीएम किसान सम्मान निधि के लिए पंजीकरण कैसे करें?",1),
 ("Which scheme gives free health insurance cover to poor families?","गरीब परिवारों को मुफ्त स्वास्थ्य बीमा देने वाली योजना कौन सी है?",2),
 ("How much hospital treatment cover does Ayushman Bharat provide?","आयुष्मान भारत में अस्पताल के इलाज के लिए कितना कवर मिलता है?",2),
 ("Which scheme helps build a pucca house for the homeless?","बेघर लोगों को पक्का मकान बनाने में कौन सी योजना मदद करती है?",3),
 ("What is the eligibility for PMAY housing subsidy?","पीएमएवाई आवास सब्सिडी के लिए पात्रता क्या है?",3),
 ("Where can a small shopkeeper get a collateral-free business loan?","छोटा दुकानदार बिना गारंटी के व्यापार ऋण कहाँ से ले सकता है?",4),
 ("What are the Shishu, Kishor and Tarun loan categories?","शिशु, किशोर और तरुण ऋण श्रेणियाँ क्या हैं?",4),
 ("Which savings scheme is meant for the education and marriage of a girl child?","बालिका की शिक्षा और विवाह के लिए बचत योजना कौन सी है?",5),
 ("What is the minimum yearly deposit in Sukanya Samriddhi?","सुकन्या समृद्धि में न्यूनतम वार्षिक जमा राशि कितनी है?",5),
 ("Which scholarship prevents dropout after class 8 for meritorious poor students?","कक्षा 8 के बाद मेधावी गरीब छात्रों की पढ़ाई छूटने से रोकने वाली छात्रवृत्ति कौन सी है?",6),
 ("What is the NMMSS scholarship amount per year?","एनएमएमएसएस छात्रवृत्ति की वार्षिक राशि कितनी है?",6),
 ("Is there a scheme for artisans and craftsmen with tool kit and training?","क्या कारीगरों और शिल्पकारों के लिए टूलकिट और प्रशिक्षण वाली योजना है?",7),
 ("Who can register under PM Vishwakarma?","पीएम विश्वकर्मा के अंतर्गत कौन पंजीकरण कर सकता है?",7),
 ("Which pension plan is for unorganised sector workers?","असंगठित क्षेत्र के कामगारों के लिए पेंशन योजना कौन सी है?",8),
 ("What guaranteed monthly pension does Atal Pension Yojana give?","अटल पेंशन योजना में गारंटीकृत मासिक पेंशन कितनी मिलती है?",8),
 ("Which state scheme gives cash transfers to a girl from birth to graduation?","कौन सी राज्य योजना बालिका को जन्म से स्नातक तक नकद सहायता देती है?",9),
 ("What documents are needed for Kanya Sumangala Yojana?","कन्या सुमंगला योजना के लिए कौन से दस्तावेज़ चाहिए?",9),
]
df = core.load_schemes_df("schemes.csv").reset_index(drop=True)
chunks = core.schemes_to_chunks(df)
td = tempfile.mkdtemp(); idx = core.VectorIndex("schemes_eval", emb, persist_path=td); idx.add(chunks)
def scheme_of(h): return [i for i,r in df.iterrows() if h.meta["source"] in (r.scheme_name, r.scheme_name_hindi)][0]
def metrics(rank_lists):
    n=len(rank_lists); r=lambda k: sum(1 for x in rank_lists if x is not None and x<=k)/n
    mrr=sum(1/x for x in rank_lists if x)/n
    return dict(n=n,hit1=r(1),hit3=r(3),hit5=r(5),mrr=mrr)
def rank_scheme(q, gold):
    hits = idx.query(q, top_k=20); seen=[]
    for h in hits:
        s=scheme_of(h)
        if s not in seen: seen.append(s)
    return seen.index(gold)+1 if gold in seen else None
res={}
for lang,col in (("EN",0),("HI",1)):
    res["E1_"+lang]=metrics([rank_scheme(q[col],q[2]) for q in Q])
# language of top chunk: does Hindi query hit the Hindi chunk?
hi_top=[idx.query(q[1],top_k=1)[0].meta["page"].startswith("दिशा") for q in Q]
en_top=[idx.query(q[0],top_k=1)[0].meta["page"].startswith("दिशा") for q in Q]
res["E1_top1_is_hindi_chunk_for_HI_queries"]=sum(hi_top)/len(Q); res["E1_top1_is_hindi_chunk_for_EN_queries"]=sum(en_top)/len(Q)

# ---------- E2: page-level retrieval on synthetic user PDFs
GT=json.load(open(HERE+"ground_truth.json")); DOCS=HERE+"docs/"
uc=[]; skipped={}
for f in sorted(os.listdir(DOCS)):
    if f.endswith(".pdf"):
        c,s=core.pdf_to_chunks(DOCS+f,f); uc+=c
        if s: skipped[f]=s
res["E2_chunks"]=len(uc); res["E2_skipped_pages"]=skipped
uidx=core.VectorIndex("user_eval",emb); uidx.add(uc)
def doc_rank(q,truth,idxs,k=20):
    hits=core.retrieve(idxs,q,top_k=k)
    for rk,h in enumerate(hits,1):
        if h.meta.get("kind")=="user_doc" and [h.meta["source"],int(str(h.meta["page"]).split()[-1])] in truth: return rk
    return None
res["E2_page_level_user_index_only"]=metrics([doc_rank(x["q"],x["truth"],[uidx]) for x in GT["answerable"]])
sidx=idx
res["E2_page_level_with_schemes_index"]=metrics([doc_rank(x["q"],x["truth"],[uidx,sidx]) for x in GT["answerable"]])
# page label format
res["E2_example_page_label"]=uc[0].meta["page"]

# ---------- E3: similarity-score separation (top-1 score, answerable vs absent)
top=lambda q,idxs: core.retrieve(idxs,q,top_k=20)[0].meta["score"]
a=[top(x["q"],[uidx]) for x in GT["answerable"]]; b=[top(q,[uidx]) for q in GT["absent"]]
res["E3_auroc_user_index"]=roc_auc_score([1]*len(a)+[0]*len(b),a+b)
res["E3_mean_top1_answerable"]=st.mean(a); res["E3_mean_top1_absent"]=st.mean(b)
res["E3_min_answerable"]=min(a); res["E3_max_absent"]=max(b)
best=max(((sum(1 for x in a if x>=t)+sum(1 for y in b if y<t))/(len(a)+len(b)),t) for t in sorted(a+b))
res["E3_best_threshold_accuracy"]=best[0]; res["E3_best_threshold"]=best[1]

# ---------- E4: latency (retrieval only, warm)
qs=[x["q"] for x in GT["answerable"]]; core.retrieve([uidx,sidx],qs[0])
lat=[]
for q in qs:
    t=time.perf_counter(); core.retrieve([uidx,sidx],q); lat.append((time.perf_counter()-t)*1000)
res["E4_retrieval_ms_median"]=st.median(lat); res["E4_retrieval_ms_p95"]=sorted(lat)[int(.95*len(lat))-1]
t=time.perf_counter(); i2=core.VectorIndex("timing_eval",emb); i2.add(uc); res["E4_index_build_s_for_user_docs"]=time.perf_counter()-t
res["backend_nominal"]=emb.backend; res["backend_actual"]=ACTUAL; res["scheme_chunks"]=len(chunks)
json.dump(res,open(HERE+"results_"+ACTUAL+os.environ.get("PYTHONHASHSEED","")+".json","w"),indent=1,ensure_ascii=False); print(json.dumps(res,indent=1,ensure_ascii=False))
