# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss
# from datetime import datetime
# import difflib

# # ================================
# # CONFIG
# # ================================
# USE_QDRANT = True
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http import models as qmodels
# except Exception:
#     USE_QDRANT = False

# # ================================
# # LLM (QUERY REWRITER ONLY)
# # ================================
# from langchain_ollama import ChatOllama

# try:
#     rewriter_llm = ChatOllama(
#         model="llama3:70b",
#         base_url="http://localhost:11434",
#         temperature=0.3
#     )

#     rewriter_llm.invoke("ping")
#     print(" Ollama is running")

# except Exception as e:
#     print(" Ollama is not running")


# # ================================
# # REGEX
# # ================================
# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# # ================================
# # HELPERS
# # ================================
# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()

# def normalize_confidence(scores, min_conf=50, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf] * len(scores)
#     return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]



# #########
# BASE_YEAR_PATTERN = re.compile(r"(20\d{2})")

# def detect_base_year(query):
#     q = query.lower()

#     if "base year" or " base" in q:
#         m = BASE_YEAR_PATTERN.search(q)
#         if m:
#             return int(m.group(1))

#     return None


# # def resolve_cpi_conflict(results, query):
# #     """
# #     Only when CPI and CPI2 both present in top results
# #     """
# #     datasets = [r["parent"] for r in results]

# #     if "CPI" not in datasets or "CPI2" not in datasets:
# #         return results  # kuch mat chhedo

# #     base_year = detect_base_year(query)

# #     # ---------- case 1: user ne base year bola ----------
# #     if base_year:
# #         if base_year >= 2024:
# #             # CPI2 rakho
# #             return [r for r in results if r["parent"] != "CPI"]
# #         else:
# #             # CPI rakho
# #             return [r for r in results if r["parent"] != "CPI2"]

# #     # ---------- case 2: base year nahi bola ----------
# #     return [r for r in results if r["parent"] != "CPI"]



# #################### new ###############

# def extract_cpi_intent(query: str):
#     prompt = f"""
# You are an intent classifier for CPI datasets.

# Query: {query}

# Return JSON only with keys:
# cpi_intent: true/false
# base_year: number or null
# wants_back_series: true/false
# has_year: number or null

# Rules:
# - CPI intent includes: CPI, inflation, price index
# - If user mentions base year → fill base_year
# - If user mentions past historical/back → wants_back_series true
# - If user mentions a year like 2021 → has_year = 2021
# """

#     try:
#         res = rewriter_llm.invoke(prompt).content.strip()
#         return json.loads(res)
#     except:
#         return {
#             "cpi_intent": False,
#             "base_year": None,
#             "wants_back_series": False,
#             "has_year": None
#         }


# def resolve_cpi_conflict(results, query):

#     intent = extract_cpi_intent(query)

#     # run only if CPI intent
#     if not intent["cpi_intent"]:
#         return results

#     datasets = [r["parent"] for r in results]
#     if "CPI" not in datasets or "CPI2" not in datasets:
#         return results

#     base_year = intent["base_year"]
#     year = intent["has_year"]
#     wants_back = intent["wants_back_series"]

#     # -------------------------------------------------
#     # 1️⃣ explicit base year mentioned
#     # -------------------------------------------------
#     if base_year:
#         if base_year >= 2024:
#             # new base → CPI2
#             return [r for r in results if r["parent"] != "CPI"]
#         else:
#             # old base → CPI
#             return [r for r in results if r["parent"] != "CPI2"]

#     # -------------------------------------------------
#     # 2️⃣ explicit back series intent
#     # -------------------------------------------------
#     if wants_back:
#         # always CPI2 back
#         return [r for r in results if r["parent"] != "CPI"]

#     # -------------------------------------------------
#     # 3️⃣ user mentioned year but NOT base year
#     # -------------------------------------------------
#     if year:

#         # year >= 2024 → new CPI2 current
#         if year >= 2024:
#             return [r for r in results if r["parent"] != "CPI"]

#         # year < 2024 → CPI (2012 base)
#         return [r for r in results if r["parent"] != "CPI2"]

#     # -------------------------------------------------
#     # 4️⃣ generic inflation query
#     # -------------------------------------------------
#     # default = latest CPI2
#     return [r for r in results if r["parent"] != "CPI"]





# # ================================
# # LLM QUERY REWRITE
# # ================================
# def rewrite_query_with_llm(user_query):
#     prompt =  f"""
# You are a QUERY NORMALIZATION ENGINE for a data analytics system.

# Task:
# Rewrite the user query safely with controlled semantic normalization.

# STRICT RULES:
# 1. DO NOT add any new information
# 2. DO NOT infer missing filters
# 3. DO NOT assume any category
# 4. DO NOT enrich meaning
# 5. ONLY rewrite words that already exist in the query
# 6. NEVER inject new concepts
# 7. NEVER add sector/gender/state unless explicitly present
# 8. Output ONLY rewritten query
# 9. No explanation
# 10. If the query contains a known dataset short form (CPI, IIP, NAS, PLFS, ASI, HCES, NSS), append its full form in the rewritten query while keeping the short form unchanged (e.g., "CPI" → "CPI Consumer Price Index"), and do not expand anything not explicitly present.
# 11. Do not remove any words from the user query


# SPECIAL RULE (VERY IMPORTANT):

# If the query contains "IIP" and also contains any month name 
# (January–December or short forms like Jan, Feb, etc.), 
# then add the word "monthly" to the query.

# Examples:
# "IIP July data" → "IIP monthly July data"
# "IIP for December" → "IIP monthly December"
# "IIP Aug 2022" → "IIP monthly Aug 2022"

# DO NOT apply this rule to any other dataset.
# If query is about CPI, GDP, PLFS etc → do nothing.


# ALLOWED OPERATIONS:
# - spelling correction
# - grammar correction
# - casing normalization
# - synonym normalization
# - semantic mapping ONLY if the word exists explicitly in text

# CRITICAL RULE (VERY IMPORTANT):
# - If the user query is ONLY a dataset or product name
#   (examples: IIP, CPI, CPIALRL, HCES, ASI,NAS, PLFS,CPI2,ASI,),
#   then RETURN THE QUERY EXACTLY AS IT IS.
# - Dataset names must NEVER be replaced with normal English words.


# STRICT SEMANTIC MAP (ONLY IF WORD EXISTS):
# - gao, gaon, village → rural
# - shehar, city, metro → urban
# - purush, aadmi, mard, man, men → male
# - mahila, aurat, lady, women → female
# - ladka → male
# - ladki → female

#  FORBIDDEN:
# - Do NOT infer urban from city names
# - Do NOT infer rural from state names
# - Do NOT infer gender from profession
# - Do NOT infer sector from geography
# - Do NOT add any category automatically

# Examples:
# RAW: "mens judge in village"
# → "male judge in rural"

# RAW: "Gini Coefficient for urban india in 2023-24"
# → "Gini Coefficient for urban in 2023-24"

# RAW: "factory output gujrat 2022"
# → "factory output Gujarat 2022"

# RAW: "men judges in delhi"
# → "male judges in Delhi"

# RAW: "factory output in gujrat for 2022 in gao"
# → "factory output in Gujarat for 2022 in rural"

# RAW: "data for mahila workers"
# → "data for female workers"

# RAW: "gaon ke factory worker"
# → "rural factory worker"

# RAW: "factory output in mumbai"
# → "factory output in Mumbai"

# User Query:
# "{user_query}"
# """
#     try:
#         out = rewriter_llm.invoke(prompt).content.strip()
#         out = out.replace('"', '').replace("\n", " ").strip()
#         return out
#     except:
#         return user_query

# # ================================
# # YEAR NORMALIZATION
# # ================================
# def normalize_year_string(s):
#     return re.sub(r"[^0-9]", "", str(s))


# def map_year_to_option(user_year, options):
#     y = int(user_year)
#     targets = [
#         f"{y}{y+1}",
#         f"{y-1}{y}",
#         str(y)
#     ]
#     norm_options = {normalize_year_string(o["option"]): o for o in options}
#     for t in targets:
#         if t in norm_options:
#             return norm_options[t]
#     return None

# # ================================
# # UNIVERSAL FILTER NORMALIZER
# # ================================
# def universal_filter_normalizer(ind_code, filters_json):
#     flat = []
#     def recurse(key, value):
#         if isinstance(value, list) and all(isinstance(x, str) for x in value):
#             for opt in value:
#                 flat.append({"parent": ind_code,"filter_name": key,"option": opt})
#         elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
#             for item in value:
#                 for k, v in item.items():
#                     if k.lower() in ["name", "title", "label"]:
#                         flat.append({"parent": ind_code,"filter_name": key,"option": v})
#                     else:
#                         recurse(k, v)
#         elif isinstance(value, dict):
#             for k, v in value.items():
#                 recurse(k, v)

#     for f in filters_json:
#         if isinstance(f, dict):
#             for k, v in f.items():
#                 recurse(k, v)
#     return flat


# #############LLM 
# # ================================
# # SMART FILTER ENGINE
# # ================================
# def select_best_filter_option(query, filter_name, options, cross_encoder):
#     q_lower = query.lower()
#     fname_lower = filter_name.lower()

#     # =========================
#     # YEAR FILTER
#     # =========================
#     if "year" in fname_lower and "base" not in fname_lower:
#         year_match = YEAR_PATTERN.search(q_lower)

#         # user ne year nahi bola → Select All
#         if not year_match:
#             return {
#                 "parent": options[0]["parent"],
#                 "filter_name": filter_name,
#                 "option": "Select All"
#             }

#         user_year = year_match.group(1)

#         mapped = map_year_to_option(user_year, options)
#         if mapped:
#             return mapped

#         pairs = [(query, f"{filter_name} {o['option']}") for o in options]
#         scores = cross_encoder.predict(pairs)
#         return options[int(np.argmax(scores))]

#     # =========================
#     # BASE YEAR FILTER (FINAL FIX)
#     # =========================
#     if "base" in fname_lower and "year" in fname_lower:

#         # 🔹 check if user explicitly mentioned base year
#         for opt in options:
#             opt_text = str(opt["option"]).lower()
#             if opt_text in q_lower:
#                 return opt

#         # 🔹 user ne base year nahi bola → latest base year pick karo
#         def extract_start_year(opt):
#             m = re.search(r"\d{4}", str(opt["option"]))
#             return int(m.group(0)) if m else 0

#         latest = max(options, key=lambda o: extract_start_year(o))
#         return latest

#     # =========================
#     # OTHER FILTERS
#     # =========================
#     mentioned = []

#     for opt in options:
#         opt_text = str(opt.get("option", "")).lower().strip()
#         if not opt_text:
#             continue

#         if opt_text in q_lower:
#             mentioned.append(opt)
#             continue

#         for word in q_lower.split():
#             if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
#                 mentioned.append(opt)
#                 break

#     if mentioned:
#         pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
#         scores = cross_encoder.predict(pairs)
#         return mentioned[int(np.argmax(scores))]

#     return {
#         "parent": options[0]["parent"],
#         "filter_name": filter_name,
#         "option": "Select All"
#     }


# # ================================
# # LOAD PRODUCTS
# # ================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

# with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     DATASETS.append({"code": ds_name, "name": ds_name})

#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind["name"],
#             "desc": ind.get("description", ""),
#             "parent": ds_name
#         })

#         flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
#         FILTERS.extend(flat)

# print(f"[INFO] DATASETS={len(DATASETS)}, INDICATORS={len(INDICATORS)}, FILTERS={len(FILTERS)}")

# # ================================
# # MODELS
# # ================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# # ================================
# # VECTOR DB
# # ================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# COLLECTION = "indicators_collection"

# qclient = None
# faiss_index = None

# if USE_QDRANT:
#     try:
#         qclient = QdrantClient(url="http://localhost:6333")
#         if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
#             qclient.recreate_collection(
#                 collection_name=COLLECTION,
#                 vectors_config=qmodels.VectorParams(size=VECTOR_DIM,distance=qmodels.Distance.COSINE)
#             )
#         print("[INFO] Qdrant ready")
#     except Exception as e:
#         USE_QDRANT = False
#         print("[WARN] Qdrant failed, using FAISS:", e)

# names = [clean_text(i["name"]) for i in INDICATORS]
# descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

# embeddings = (0.4 * bi_encoder.encode(names, convert_to_numpy=True) + 0.6 * bi_encoder.encode(descs, convert_to_numpy=True))
# embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# if USE_QDRANT and qclient:
#     qclient.upsert(
#         collection_name=COLLECTION,
#         points=[qmodels.PointStruct(id=i,vector=embeddings[i].tolist(),payload=INDICATORS[i]) for i in range(len(INDICATORS))]
#     )
# else:
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings.astype("float32"))

# # ================================
# # SEARCH
# # ================================
# def search_indicators(query, top_k=25, max_products=3):
#     q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
#     q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

#     if USE_QDRANT and qclient:
#         hits = qclient.search(collection_name=COLLECTION,query_vector=q_vec[0].tolist(),limit=top_k)
#         candidates = [h.payload for h in hits]
#     else:
#         _, I = faiss_index.search(q_vec.astype("float32"), top_k)
#         candidates = [INDICATORS[i] for i in I[0] if i >= 0]

#     scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
#     for i, c in enumerate(candidates):
#         c["score"] = float(scores[i])

#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     # CPI conflict resolve ONLY if both present
#     candidates = resolve_cpi_conflict(candidates, query)

#     seen, final = set(), []
#     for c in candidates:

#         if c["parent"] not in seen:
#             seen.add(c["parent"])
#             final.append(c)
#         if len(final) == max_products:
#             break


#     return final




# ###################query capture 


# import uuid
# from datetime import datetime

# LOG_FILE = os.path.join(BASE_DIR, "logs", "queries.jsonl")

# def save_query_log(raw_query, rewritten_query, response_json):
#     os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

#     record = {
#         "id": str(uuid.uuid4()),
#         "timestamp": datetime.utcnow().isoformat(),
#         "raw_query": raw_query,
#         "rewritten_query": rewritten_query,
#         "response": response_json
#     }

#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


# # ================================
# # FLASK
# # ================================
# app = Flask(__name__, template_folder="templates")
# CORS(app)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     raw_q = request.json.get("query", "").strip()
#     if not raw_q:
#         return jsonify({"error": "query required"}), 400

#     #  LLM rewrite
#     q = rewrite_query_with_llm(raw_q)

#     print("RAW :", raw_q)
#     print("LLM :", q)

#     top_results = search_indicators(q)
#     confidences = normalize_confidence([r["score"] for r in top_results])

#     results = []

#     for ind, conf in zip(top_results, confidences):
#         dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
#         related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

#         grouped = {}
#         for f in related_filters:
#             grouped.setdefault(f["filter_name"], []).append(f)

#         best_filters = []
#         for fname, opts in grouped.items():
#             best_opt = select_best_filter_option(
#                 query=q,
#                 filter_name=fname,
#                 options=opts,
#                 cross_encoder=cross_encoder
#             )
#             best_filters.append({
#                 "filter_name": fname,
#                 "option": best_opt["option"]
#             })

#         results.append({
#             "dataset": dataset["name"],
#             "indicator": ind["name"],
#             "confidence": conf,
#             "filters": best_filters
#         })
#     response = {"results": results}
#         #  SAVE OUTPUT
#     save_query_log(
#         raw_query=raw_q,
#         rewritten_query=q,
#         response_json=response
#     )

#     #return jsonify(response)

#     return jsonify({"results": results})

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5009)






# from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
# import os, json, re
# import numpy as np
# from sentence_transformers import SentenceTransformer, CrossEncoder
# import faiss
# from datetime import datetime
# import difflib

# # ================================
# # CONFIG
# # ================================
# USE_QDRANT = True
# try:
#     from qdrant_client import QdrantClient
#     from qdrant_client.http import models as qmodels
# except Exception:
#     USE_QDRANT = False

# # ================================
# # LLM (QUERY REWRITER ONLY)
# # ================================
# from langchain_ollama import ChatOllama

# try:
#     rewriter_llm = ChatOllama(
#         model="llama3:70b",
#         base_url="http://localhost:11434",
#         temperature=0.3
#     )

#     rewriter_llm.invoke("ping")
#     print(" Ollama is running")

# except Exception as e:
#     print(" Ollama is not running")


# # ================================
# # REGEX
# # ================================
# YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# # ================================
# # HELPERS
# # ================================
# def clean_text(t):
#     t = (t or "").lower()
#     t = re.sub(r"[^a-z0-9\s]", " ", t)
#     return re.sub(r"\s+", " ", t).strip()

# def normalize_confidence(scores, min_conf=50, max_conf=95):
#     if not scores:
#         return []
#     mn, mx = min(scores), max(scores)
#     if mn == mx:
#         return [min_conf] * len(scores)
#     return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]



# #########
# BASE_YEAR_PATTERN = re.compile(r"(20\d{2})")

# def detect_base_year(query):
#     q = query.lower()

#     if "base year" or " base" in q:
#         m = BASE_YEAR_PATTERN.search(q)
#         if m:
#             return int(m.group(1))

#     return None


# def resolve_cpi_conflict(results, query):
#     """
#     Only when CPI and CPI2 both present in top results
#     """
#     datasets = [r["parent"] for r in results]

#     if "CPI" not in datasets or "CPI2" not in datasets:
#         return results  # kuch mat chhedo

#     base_year = detect_base_year(query)

#     # ---------- case 1: user ne base year bola ----------
#     if base_year:
#         if base_year >= 2024:
#             # CPI2 rakho
#             return [r for r in results if r["parent"] != "CPI"]
#         else:
#             # CPI rakho
#             return [r for r in results if r["parent"] != "CPI2"]

#     # ---------- case 2: base year nahi bola ----------
#     return [r for r in results if r["parent"] != "CPI"]


# # ================================
# # LLM QUERY REWRITE
# # ================================
# def rewrite_query_with_llm(user_query):
#     prompt =  f"""
# You are a QUERY NORMALIZATION ENGINE for a data analytics system.

# Task:
# Rewrite the user query safely with controlled semantic normalization.

# STRICT RULES:
# 1. DO NOT add any new information
# 2. DO NOT infer missing filters
# 3. DO NOT assume any category
# 4. DO NOT enrich meaning
# 5. ONLY rewrite words that already exist in the query
# 6. NEVER inject new concepts
# 7. NEVER add sector/gender/state unless explicitly present
# 8. Output ONLY rewritten query
# 9. No explanation
# 10. If the query contains a known dataset short form (CPI, IIP, NAS, PLFS, ASI, HCES, NSS), append its full form in the rewritten query while keeping the short form unchanged (e.g., "CPI" → "CPI Consumer Price Index"), and do not expand anything not explicitly present.



# SPECIAL RULE (VERY IMPORTANT):

# If the query contains "IIP" and also contains any month name 
# (January–December or short forms like Jan, Feb, etc.), 
# then add the word "monthly" to the query.

# If the query contain Q1 or Q2 or Q3 or Q4 then add quarterly but do not remove Q1 or Q2 or Q3 or Q4 

# Examples:
# "IIP July data" → "IIP monthly July data"
# "IIP for December" → "IIP monthly December"
# "IIP Aug 2022" → "IIP monthly Aug 2022"

# DO NOT apply this rule to any other dataset.
# If query is about CPI, GDP, PLFS etc → do nothing.


# ALLOWED OPERATIONS:
# - spelling correction
# - grammar correction
# - casing normalization
# - synonym normalization
# - semantic mapping ONLY if the word exists explicitly in text

# CRITICAL RULE (VERY IMPORTANT):
# - If the user query is ONLY a dataset or product name
#   (examples: IIP, CPI, CPIALRL, HCES, ASI,NAS, PLFS,CPI2,ASI,),
#   then RETURN THE QUERY EXACTLY AS IT IS.
# - Dataset names must NEVER be replaced with normal English words.
# SPECIAL RULE:
# If query contains both "year" and "base year", clearly separate them:
# - "gdp for year 2023-24 base year 2022-23" → "gdp year:2023-24 base_year:2022-23"



# STRICT SEMANTIC MAP (ONLY IF WORD EXISTS):
# - gao, gaon, village → rural
# - shehar, city, metro → urban
# - purush, aadmi, mard, man, men → male
# - mahila, aurat, lady, women → female
# - ladka → male
# - ladki → female

# ❌ FORBIDDEN:
# - Do NOT infer urban from city names
# - Do NOT infer rural from state names
# - Do NOT infer gender from profession
# - Do NOT infer sector from geography
# - Do NOT add any category automatically

# Examples:
# RAW: "mens judge in village"
# → "male judge in rural"

# RAW: "Gini Coefficient for urban india in 2023-24"
# → "Gini Coefficient for urban in 2023-24"

# RAW: "factory output gujrat 2022"
# → "factory output Gujarat 2022"

# RAW: "men judges in delhi"
# → "male judges in Delhi"

# RAW: "factory output in gujrat for 2022 in gao"
# → "factory output in Gujarat for 2022 in rural"

# RAW: "data for mahila workers"
# → "data for female workers"

# RAW: "gaon ke factory worker"
# → "rural factory worker"

# RAW: "factory output in mumbai"
# → "factory output in Mumbai"

# User Query:
# "{user_query}"
# """
#     try:
#         out = rewriter_llm.invoke(prompt).content.strip()
#         out = out.replace('"', '').replace("\n", " ").strip()
#         return out
#     except:
#         return user_query

# # ================================
# # YEAR NORMALIZATION
# # ================================
# def normalize_year_string(s):
#     return re.sub(r"[^0-9]", "", str(s))


# def map_year_to_option(user_year, options):
#     y = int(user_year)
#     targets = [
#          f"{y}{y+1}",            # → "20232024"
#         f"{y}{str(y+1)[-2:]}",  # → "202324"  ← NEW!
#         f"{y-1}{y}",            # → "20222023"
#         f"{y-1}{str(y)[-2:]}",  # → "202223"  ← NEW!
#         str(y)                   # → "2023"
#     ]
#     norm_options = {normalize_year_string(o["option"]): o for o in options}
#     for t in targets:
#         if t in norm_options:
#             return norm_options[t]
#     return None

# # ================================
# # UNIVERSAL FILTER NORMALIZER
# # ================================
# def universal_filter_normalizer(ind_code, filters_json):
#     flat = []
#     def recurse(key, value):
#         if isinstance(value, list) and all(isinstance(x, str) for x in value):
#             for opt in value:
#                 flat.append({"parent": ind_code,"filter_name": key,"option": opt})
#         elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
#             for item in value:
#                 for k, v in item.items():
#                     if k.lower() in ["name", "title", "label"]:
#                         flat.append({"parent": ind_code,"filter_name": key,"option": v})
#                     else:
#                         recurse(k, v)
#         elif isinstance(value, dict):
#             for k, v in value.items():
#                 recurse(k, v)

#     for f in filters_json:
#         if isinstance(f, dict):
#             for k, v in f.items():
#                 recurse(k, v)
#     return flat


# #############LLM 
# # ================================
# # SMART FILTER ENGINE
# # ================================
# def select_best_filter_option(query, filter_name, options, cross_encoder):
#     q_lower = query.lower()
#     fname_lower = filter_name.lower()

#     # =========================
#     # YEAR FILTER
#     # =========================
#     if "year" in fname_lower and "base" not in fname_lower:
#         year_match = YEAR_PATTERN.search(q_lower)
#         use_year=year_match.group(1)
#         mapped=map_year_to_option(use_year,options)

#         # user ne year nahi bola → Select All
#         if not year_match:
#             return {
#                 "parent": options[0]["parent"],
#                 "filter_name": filter_name,
#                 "option": "Select All"
#             }

#         user_year = year_match.group(1)

#         mapped = map_year_to_option(user_year, options)
#         if mapped:
#             return mapped

#         pairs = [(query, f"{filter_name} {o['option']}") for o in options]
#         scores = cross_encoder.predict(pairs)
#         return options[int(np.argmax(scores))]

#     # =========================
#     # BASE YEAR FILTER (FINAL FIX)
#     # =========================
#     if "base" in fname_lower and "year" in fname_lower:

#         # 🔹 check if user explicitly mentioned base year
#         for opt in options:
#             opt_text = str(opt["option"]).lower()
#             if opt_text in q_lower:
#                 return opt

#         # 🔹 user ne base year nahi bola → latest base year pick karo
#         def extract_start_year(opt):
#             m = re.search(r"\d{4}", str(opt["option"]))
#             return int(m.group(0)) if m else 0

#         latest = max(options, key=lambda o: extract_start_year(o))
#         return latest

#     # =========================
#     # OTHER FILTERS
#     # =========================
#     mentioned = []

#     for opt in options:
#         opt_text = str(opt.get("option", "")).lower().strip()
#         if not opt_text:
#             continue

#         if opt_text in q_lower:
#             mentioned.append(opt)
#             continue

#         for word in q_lower.split():
#             if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.70:
#                 mentioned.append(opt)
#                 break

#     if mentioned:
#         pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
#         scores = cross_encoder.predict(pairs)
#         return mentioned[int(np.argmax(scores))]

#     return {
#         "parent": options[0]["parent"],
#         "filter_name": filter_name,
#         "option": "Select All"
#     }


# # ================================
# # LOAD PRODUCTS
# # ================================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")

# with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
#     raw_products = json.load(f)

# DATASETS, INDICATORS, FILTERS = [], [], []

# for ds_name, ds_info in raw_products.get("datasets", {}).items():
#     DATASETS.append({"code": ds_name, "name": ds_name})

#     for ind in ds_info.get("indicators", []):
#         ind_code = f"{ds_name}_{ind['name']}"
#         INDICATORS.append({
#             "code": ind_code,
#             "name": ind["name"],
#             "desc": ind.get("description", ""),
#             "parent": ds_name
#         })

#         flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
#         FILTERS.extend(flat)

# print(f"[INFO] DATASETS={len(DATASETS)}, INDICATORS={len(INDICATORS)}, FILTERS={len(FILTERS)}")

# # ================================
# # MODELS
# # ================================
# bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# # ================================
# # VECTOR DB
# # ================================
# VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
# COLLECTION = "indicators_collection"

# qclient = None
# faiss_index = None

# if USE_QDRANT:
#     try:
#         qclient = QdrantClient(url="http://localhost:6333")
#         if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
#             qclient.recreate_collection(
#                 collection_name=COLLECTION,
#                 vectors_config=qmodels.VectorParams(size=VECTOR_DIM,distance=qmodels.Distance.COSINE)
#             )
#         print("[INFO] Qdrant ready")
#     except Exception as e:
#         USE_QDRANT = False
#         print("[WARN] Qdrant failed, using FAISS:", e)

# names = [clean_text(i["name"]) for i in INDICATORS]
# descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

# embeddings = (0.4 * bi_encoder.encode(names, convert_to_numpy=True) + 0.6 * bi_encoder.encode(descs, convert_to_numpy=True))
# embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

# if USE_QDRANT and qclient:
#     qclient.upsert(
#         collection_name=COLLECTION,
#         points=[qmodels.PointStruct(id=i,vector=embeddings[i].tolist(),payload=INDICATORS[i]) for i in range(len(INDICATORS))]
#     )
# else:
#     faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
#     faiss_index.add(embeddings.astype("float32"))

# # ================================
# # SEARCH
# # ================================
# def search_indicators(query, top_k=25, max_products=3):
#     q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
#     q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

#     if USE_QDRANT and qclient:
#         hits = qclient.search(collection_name=COLLECTION,query_vector=q_vec[0].tolist(),limit=top_k)
#         candidates = [h.payload for h in hits]
#     else:
#         _, I = faiss_index.search(q_vec.astype("float32"), top_k)
#         candidates = [INDICATORS[i] for i in I[0] if i >= 0]

#     scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
#     for i, c in enumerate(candidates):
#         c["score"] = float(scores[i])

#     candidates.sort(key=lambda x: x["score"], reverse=True)

#     # CPI conflict resolve ONLY if both present
#     candidates = resolve_cpi_conflict(candidates, query)

#     seen, final = set(), []
#     for c in candidates:

#         if c["parent"] not in seen:
#             seen.add(c["parent"])
#             final.append(c)
#         if len(final) == max_products:
#             break


#     return final




# ###################query capture 


# import uuid
# from datetime import datetime

# LOG_FILE = os.path.join(BASE_DIR, "logs", "queries.jsonl")

# def save_query_log(raw_query, rewritten_query, response_json):
#     os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

#     record = {
#         "id": str(uuid.uuid4()),
#         "timestamp": datetime.utcnow().isoformat(),
#         "raw_query": raw_query,
#         "rewritten_query": rewritten_query,
#         "response": response_json
#     }

#     with open(LOG_FILE, "a", encoding="utf-8") as f:
#         f.write(json.dumps(record, ensure_ascii=False) + "\n")


# # ================================
# # FLASK
# # ================================
# app = Flask(__name__, template_folder="templates")
# CORS(app)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     raw_q = request.json.get("query", "").strip()
#     if not raw_q:
#         return jsonify({"error": "query required"}), 400

#     #  LLM rewrite
#     q = rewrite_query_with_llm(raw_q)

#     print("RAW :", raw_q)
#     print("LLM :", q)

#     top_results = search_indicators(q)
#     confidences = normalize_confidence([r["score"] for r in top_results])

#     results = []

#     for ind, conf in zip(top_results, confidences):
#         dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
#         related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

#         grouped = {}
#         for f in related_filters:
#             grouped.setdefault(f["filter_name"], []).append(f)

#         best_filters = []
#         for fname, opts in grouped.items():
#             best_opt = select_best_filter_option(
#                 query=q,
#                 filter_name=fname,
#                 options=opts,
#                 cross_encoder=cross_encoder
#             )
#             best_filters.append({
#                 "filter_name": fname,
#                 "option": best_opt["option"]
#             })

#         results.append({
#             "dataset": dataset["name"],
#             "indicator": ind["name"],
#             "confidence": conf,
#             "filters": best_filters
#         })
#     response = {"results": results}
#         #  SAVE OUTPUT
#     save_query_log(
#         raw_query=raw_q,
#         rewritten_query=q,
#         response_json=response
#     )

#     #return jsonify(response)

#     return jsonify({"results": results})

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5009)






from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, json, re
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
from datetime import datetime
import difflib

def _agent_log(*, runId: str, hypothesisId: str, location: str, message: str, data: dict | None = None):
    return


# ================================
# CONFIG
# ================================
USE_QDRANT = True
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:
    USE_QDRANT = False

# ================================
# LLM (QUERY REWRITER ONLY)
# ================================
from langchain_ollama import ChatOllama

try:
    rewriter_llm = ChatOllama(
        model="llama3:70b",
        base_url="http://localhost:11434",
        temperature=0.3
    )

    rewriter_llm.invoke("ping")
    print(" Ollama is running")

except Exception as e:
    print(" Ollama is not running")


# ================================
# REGEX
# ================================
YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")

# ================================
# HELPERS
# ================================
def clean_text(t):
    """Text ko lowercase karke special chars hatao, sirf a-z 0-9 space rakho. Embedding/search ke liye normalize."""
    t = (t or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def normalize_confidence(scores, min_conf=50, max_conf=95):
    """Scores ko min_conf se max_conf range mein scale karo. Sab indicators ko 50-95% confidence range mein map."""
    if not scores:
        return []
    mn, mx = min(scores), max(scores)
    if mn == mx:
        return [min_conf] * len(scores)
    return [round(min_conf + (s - mn)/(mx - mn)*(max_conf - min_conf), 2) for s in scores]



#########
BASE_YEAR_PATTERN = re.compile(r"(20\d{2})")

def detect_base_year(query):
    """Query mein base year (20xx) detect karo. CPI/CPI2 conflict resolve ke liye use hota hai."""
    q = query.lower()

    if "base year" or " base" in q:
        m = BASE_YEAR_PATTERN.search(q)
        if m:
            return int(m.group(1))

    return None


def resolve_cpi_conflict(results, query):
    """Jab CPI aur CPI2 dono top results mein hon: base year 2024+ → CPI2 rakho, else CPI rakho. Default: CPI2."""
    # Only when CPI and CPI2 both present in top results
    datasets = [r["parent"] for r in results]

    if "CPI" not in datasets or "CPI2" not in datasets:
        return results  # kuch mat chhedo

    base_year = detect_base_year(query)

    # ---------- case 1: user ne base year bola ----------
    if base_year:
        if base_year >= 2024:
            # CPI2 rakho
            return [r for r in results if r["parent"] != "CPI"]
        else:
            # CPI rakho
            return [r for r in results if r["parent"] != "CPI2"]

    # ---------- case 2: base year nahi bola ----------
    return [r for r in results if r["parent"] != "CPI"]


# ================================
# CPI INDICATOR KEYWORD MAPPING (golden rule: CPI/CPI2 only)
# ================================
# Query phrases -> preferred indicator name substring (in name or desc)
_CPI_QUERY_TO_INDICATOR = [
    ("general inflation", "General Index"),
    ("general index", "General Index"),
    ("combined inflation", "General Index"),
    ("combined general", "General Index"),
    ("headline inflation", "General Index"),
    ("all india inflation", "General Index"),
    ("housing", "Group"),
    ("urban index", "Group"),
    ("rural index", "Group"),
    ("fuel and light", "Group"),
    ("food and beverage", "Group"),
    ("food and beverages", "Group"),
    ("group", "Group"),
    ("item", "Item"),
    ("rice", "Item"),
    ("mustard", "Item"),
    ("vegetable", "Item"),
    ("pds", "Item"),
    ("subgroup", "Subgroup"),
    ("sub group", "Subgroup"),
]


def _choose_best_cpi_indicator(pool, query):
    """
    CPI/CPI2 candidates mein se query ke hisaab se best indicator choose karo.
    Golden rule: sirf CPI/CPI2 ke liye; baaki datasets pe no-op (caller only passes CPI/CPI2 pool).
    """
    if not pool:
        return None
    q = (query or "").lower()
    best = None
    best_score = -1.0
    for c in pool:
        s = float(c.get("score", 0))
        name_l = (c.get("name") or "").lower()
        desc_l = (c.get("desc") or "").lower()
        combined = name_l + " " + desc_l
        for phrase, ind_name in _CPI_QUERY_TO_INDICATOR:
            if phrase in q and (ind_name.lower() in combined):
                s += 1.0
                break
        if s > best_score:
            best_score = s
            best = c
    return best if best else pool[0]


def _reorder_final_by_cpi_cpialrl_wpi_intent(final, query):
    """
    Jab final list mein CPI/CPI2/CPIALRL/WPI ho, query intent ke hisaab se order fix karo:
    wholesale/wpi -> WPI first; agricultural labour/cpialrl -> CPIALRL first; retail/inflation/cpi -> CPI/CPI2 first.
    Golden rule: sirf order change; koi dataset add/remove nahi.
    """
    if not final:
        return final
    parents = [r.get("parent") for r in final]
    if not any(p in ("CPI", "CPI2", "CPIALRL", "WPI") for p in parents):
        return final
    q = (query or "").lower()

    def _intent_priority(r):
        p = r.get("parent")
        if p == "WPI":
            return 0 if ("wholesale" in q or "wpi" in q or "factory price" in q) else 2
        if p == "CPIALRL":
            return 0 if (
                "agricultural" in q or "rural labour" in q or "cpialrl" in q
                or "agricultural labour" in q or "agri labour" in q
            ) else 2
        if p in ("CPI", "CPI2"):
            return 0 if (
                "inflation" in q or "retail" in q or "consumer price" in q
                or "general" in q or "cpi" in q or "price index" in q
            ) and "wholesale" not in q else 1
        return 2

    def _score_val(r):
        try:
            return -float(r.get("score") or 0)
        except (TypeError, ValueError):
            return 0
    return sorted(final, key=lambda r: (_intent_priority(r), _score_val(r)))


# ================================
# LLM QUERY REWRITE
# ================================
def rewrite_query_with_llm(user_query):
    """Ollama LLM se query normalize/rewrite karo (spelling, synonyms, dataset full form). Fail → raw query return."""
    prompt =  f"""
You are a QUERY NORMALIZATION ENGINE for a data analytics system.

Task:
Rewrite the user query safely with controlled semantic normalization.

STRICT RULES:
1. DO NOT add any new information
2. DO NOT infer missing filters
3. DO NOT assume any category
4. DO NOT enrich meaning
5. ONLY rewrite words that already exist in the query
6. NEVER inject new concepts
7. NEVER add sector/gender/state unless explicitly present
8. Output ONLY rewritten query
9. No explanation
10. If the query contains a known dataset short form (CPI, IIP, NAS, PLFS, ASI, HCES, NSS, EC, WPI), append its full form in the rewritten query while keeping the short form unchanged (e.g., "CPI" → "CPI Consumer Price Index", "EC" → "EC Economic Census", "WPI" → "WPI Wholesale Price Index"), and do not expand anything not explicitly present.



SPECIAL RULE (VERY IMPORTANT):

If the query contains "IIP" and also contains any month name 
(January–December or short forms like Jan, Feb, etc.), 
then add the word "monthly" to the query.

If query contains both "year" and "base year", clearly separate them:


Examples:
"IIP July data" → "IIP monthly July data"
"IIP for December" → "IIP monthly December"
"IIP Aug 2022" → "IIP monthly Aug 2022"
"gdp for year 2023-24 base year 2022-23" → "gdp year:2023-24 base_year:2022-23"

DO NOT apply this rule to any other dataset.
If query is about CPI, GDP, PLFS etc → do nothing.


ALLOWED OPERATIONS:
- spelling correction
- grammar correction
- casing normalization
- synonym normalization
- semantic mapping ONLY if the word exists explicitly in text

CRITICAL RULE (VERY IMPORTANT):
- If the user query is ONLY a dataset or product name
  (examples: IIP, CPI, CPIALRL, HCES, ASI, NAS, PLFS, CPI2, EC, EC4, EC5, EC6, WPI),
  then: "EC" → "EC Economic Census" (matches EC4/EC5/EC6); "WPI" → "WPI Wholesale Price Index"; others RETURN AS IS.
- Dataset names must NEVER be replaced with normal English words.


STRICT SEMANTIC MAP (ONLY IF WORD EXISTS):
- gao, gaon, village → rural
- shehar, city, metro → urban
- purush, aadmi, mard, man, men → male
- mahila, aurat, lady, women → female
- ladka → male
- ladki → female

❌ FORBIDDEN:
- Do NOT infer urban from city names
- Do NOT infer rural from state names
- Do NOT infer gender from profession
- Do NOT infer sector from geography
- Do NOT add any category automatically

Examples:
RAW: "mens judge in village"
→ "male judge in rural"

RAW: "Gini Coefficient for urban india in 2023-24"
→ "Gini Coefficient for urban in 2023-24"

RAW: "factory output gujrat 2022"
→ "factory output Gujarat 2022"

RAW: "men judges in delhi"
→ "male judges in Delhi"

RAW: "factory output in gujrat for 2022 in gao"
→ "factory output in Gujarat for 2022 in rural"

RAW: "data for mahila workers"
→ "data for female workers"

RAW: "gaon ke factory worker"
→ "rural factory worker"

RAW: "factory output in mumbai"
→ "factory output in Mumbai"

User Query:
"{user_query}"
"""
    try:
        out = rewriter_llm.invoke(prompt).content.strip()
        out = out.replace('"', '').replace("\n", " ").strip()
        return out
    except:
        return user_query

# ================================
# YEAR NORMALIZATION
# ================================
def normalize_year_string(s):
    """String se sirf digits nikalo (e.g. '2023-24' → '202324'). Year matching ke liye."""
    return re.sub(r"[^0-9]", "", str(s))


def map_year_to_option(user_year, options):
    """User year (e.g. 2023) ko options (2023-24, 2022-23, etc.) mein map karo. Match nahi → None."""
    y = int(user_year)
    targets = [
         f"{y}{y+1}",            # → "20232024"
        f"{y}{str(y+1)[-2:]}",  # → "202324"  ← NEW!
        f"{y-1}{y}",            # → "20222023"
        f"{y-1}{str(y)[-2:]}",  # → "202223"  ← NEW!
        str(y)                   # → "2023"
    ]
    norm_options = {normalize_year_string(o["option"]): o for o in options}
    for t in targets:
        if t in norm_options:
            return norm_options[t]
    return None

# ================================
# FILTER ACCURACY & ESSENTIAL FILTERS (Moth criteria)
# Filter Accuracy = 4 filters only: Year, Sector, Gender, State (when present, must appear first)
# Essential Filters Accuracy = CPI: Series, Base Year; IIP: Base Year; ASI: Classification Year;
#                             NAS: Series, Frequency; CPIALRL: Base Year (when present, must appear)
# ================================
# 4 filters - Filter Accuracy basis
MANDATORY_4 = ["Year", "Sector", "Gender", "State"]

# Essential filters per dataset - Essential Filters Accuracy basis
ESSENTIAL_FILTERS_BY_DATASET = {
    "CPI": ["Series", "Base_Year"],
    "IIP": ["Base_Year"],
    "ASI": ["classification_year"],
    "NAS": ["Series", "Frequency"],
    "CPIALRL": ["Base_Year"],
}


def _priority_order_for_dataset(parent_code):
    """Filter ka priority order banao: Year,Sector,Gender,State pehle, phir dataset essential (Series,Base_Year,etc.), phir rest."""
    order = ["Year", "financial_Year", "Sector", "Gender", "State"]
    essential = ESSENTIAL_FILTERS_BY_DATASET.get(parent_code, [])
    for e in essential:
        if e not in order:
            order.append(e)
    for k in ["Base_Year", "Series", "classification_year", "Frequency"]:
        if k not in order:
            order.append(k)
    return order


def ensure_mandatory_filter_order(best_filters, parent_code):
    """best_filters ko Moth criteria ke hisaab se reorder: 4 filters first, phir essential, phir baaki. Sirf reorder, kuch add/remove nahi."""
    if not best_filters:
        return best_filters
    by_name = {f["filter_name"]: f for f in best_filters}
    ordered = []
    priority = _priority_order_for_dataset(parent_code)
    for key in priority:
        if key in by_name:
            ordered.append(by_name.pop(key))
    for f in best_filters:
        if f["filter_name"] in by_name:
            ordered.append(by_name.pop(f["filter_name"]))
    for v in by_name.values():
        ordered.append(v)
    return ordered


def ensure_required_filters_present(best_filters, parent_code, grouped, query, cross_encoder, hints: dict | None = None):
    """Jo required filters (4 + essential) grouped mein hain lekin best_filters mein nahi, unko add karo. Phir sahi order apply karo."""
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="B",
        location="running_code.py:ensure_required_filters_present:entry",
        message="Ensuring mandatory+essential filters are present",
        data={
            "parent_code": parent_code,
            "initial_filters": [f.get("filter_name") for f in (best_filters or [])],
            "grouped_keys": list((grouped or {}).keys())[:80],
        },
    )
    # endregion agent log
    required = list(MANDATORY_4)
    required.extend(ESSENTIAL_FILTERS_BY_DATASET.get(parent_code, []))
    required = list(dict.fromkeys(required))
    out_names = {f["filter_name"]: f for f in best_filters}
    added = []
    for r in required:
        if r in out_names:
            continue
        if r not in grouped:
            continue
        opts = grouped[r]
        if not opts:
            continue
        best_opt = select_best_filter_option(query, r, opts, cross_encoder, hints=hints)
        best_filters.append({"filter_name": r, "option": best_opt["option"]})
        out_names[r] = best_filters[-1]
        added.append(r)
    best_filters = ensure_mandatory_filter_order(best_filters, parent_code)
    # Golden rule: CPI-only (do not affect CPI2 / other datasets)
    best_filters = ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query)
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="B",
        location="running_code.py:ensure_required_filters_present:exit",
        message="Required filters ensured and ordered",
        data={
            "parent_code": parent_code,
            "added": added,
            "final_filters": [f.get("filter_name") for f in (best_filters or [])],
        },
    )
    # endregion agent log
    return best_filters


def ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query):
    """CPI ke liye Series+Base_Year valid combo ensure karo (Current↔2012, Back↔2010 unless user explicitly says otherwise)."""
    # Golden rule: narrow scope only
    if parent_code != "CPI":
        return best_filters
    by_name = {f["filter_name"]: f for f in best_filters}
    if "Series" not in by_name or "Base_Year" not in by_name:
        return best_filters
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="C",
        location="running_code.py:ensure_cpi_series_base_year_consistent:before",
        message="CPI Series/Base_Year consistency check",
        data={
            "parent_code": parent_code,
            "query_snippet": (query or "")[:120],
            "series_before": by_name.get("Series", {}).get("option"),
            "base_year_before": by_name.get("Base_Year", {}).get("option"),
            "series_opts_n": len((grouped or {}).get("Series", []) or []),
            "base_opts_n": len((grouped or {}).get("Base_Year", []) or []),
        },
    )
    # endregion agent log
    q_lower = query.lower()
    series_opt = str(by_name["Series"].get("option", "")).lower()
    base_opt = str(by_name["Base_Year"].get("option", "")).lower()
    base_year = re.search(r"20\d{2}", base_opt)
    base_year = base_year.group(0) if base_year else ""
    target_series, target_base = None, None
    # If query explicitly mentions base year 2010/2012, respect it; otherwise apply default linkage.
    if re.search(r"\bbase\s*year\s*2010\b", q_lower) or re.search(r"\b2010\s*base\b", q_lower):
        target_series, target_base = "Back", "2010"
    elif re.search(r"\bbase\s*year\s*2012\b", q_lower) or re.search(r"\b2012\s*base\b", q_lower):
        target_series, target_base = "Current", "2012"
    elif "back" in q_lower or "back series" in q_lower or "historical" in q_lower:
        target_series, target_base = "Back", "2010"
    elif "current" in q_lower:
        target_series, target_base = "Current", "2012"
    elif series_opt == "back" and base_year and base_year != "2010":
        target_series, target_base = "Back", "2010"
    elif series_opt == "current" and base_year and base_year != "2012":
        target_series, target_base = "Current", "2012"
    elif base_year == "2010" and series_opt != "back":
        target_series, target_base = "Back", "2010"
    elif base_year == "2012" and series_opt != "current":
        target_series, target_base = "Current", "2012"
    elif not base_year and series_opt == "current":
        target_base = "2012"
    elif not base_year and series_opt == "back":
        target_base = "2010"
    if not target_series and not target_base:
        return best_filters
    series_opts = grouped.get("Series", [])
    base_opts = grouped.get("Base_Year", [])
    if target_series:
        for opt in series_opts:
            if str(opt.get("option", "")).lower() == target_series.lower():
                by_name["Series"]["option"] = opt["option"]
                break
    if target_base:
        for opt in base_opts:
            if target_base in str(opt.get("option", "")):
                by_name["Base_Year"]["option"] = opt["option"]
                break
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="C",
        location="running_code.py:ensure_cpi_series_base_year_consistent:after",
        message="CPI Series/Base_Year corrected (if needed)",
        data={
            "parent_code": parent_code,
            "target_series": target_series,
            "target_base": target_base,
            "series_after": by_name.get("Series", {}).get("option"),
            "base_year_after": by_name.get("Base_Year", {}).get("option"),
        },
    )
    # endregion agent log
    return best_filters


# ================================
# UNIVERSAL FILTER NORMALIZER
# ================================
def universal_filter_normalizer(ind_code, filters_json):
    """products.json ke nested filters ko flat list mein convert karo: [{parent, filter_name, option}, ...]. Nested dict/list recurse karta hai."""
    flat = []
    def recurse(key, value):
        if isinstance(value, list) and all(isinstance(x, str) for x in value):
            for opt in value:
                flat.append({"parent": ind_code,"filter_name": key,"option": opt})
        elif isinstance(value, list) and all(isinstance(x, dict) for x in value):
            for item in value:
                for k, v in item.items():
                    if k.lower() in ["name", "title", "label"]:
                        flat.append({"parent": ind_code,"filter_name": key,"option": v})
                    else:
                        recurse(k, v)
        elif isinstance(value, dict):
            for k, v in value.items():
                recurse(k, v)

    for f in filters_json:
        if isinstance(f, dict):
            for k, v in f.items():
                recurse(k, v)
    return flat


#############LLM 
# ================================
# SMART FILTER ENGINE
# ================================
def select_best_filter_option(query, filter_name, options, cross_encoder, hints: dict | None = None):
    """
    Query ke hisaab se sabse sahi filter option pick karo.
    Optional `hints` (dataset-specific intent layer) se pehle direct match try karta hai.
    """
    if not options:
        return {"parent": "", "filter_name": filter_name, "option": "Select All"}
    q_lower = (query or "").lower()
    fname_lower = filter_name.lower()
    hints = hints or {}
    hint_value = hints.get(filter_name)
     
    # =========================
    # FREQUENCY FILTER
    # =========================
    if fname_lower in ["frequency"]:
        # --- Hint se frequency decide ho chuki hai (NAS / ENVSTAT / TUS / PLFS etc.) ---
        if isinstance(hint_value, str):
            hv = hint_value.lower()
            for opt in options:
                o = str(opt.get("option", "")).lower()
                if o.startswith(hv) or hv.startswith(o):
                    return opt

        # --- Check for explicit mention ---
        for keyword in ["annually", "quarterly", "monthly", "annual"]:
            if keyword in q_lower:
                for opt in options:
                    o = str(opt.get("option", "")).lower()
                    if o.startswith(keyword) or keyword.startswith(o):
                        return opt

        # --- Month names → Monthly (full names only to avoid "may" false positive) ---
        month_names = [
            "january", "february", "march", "april", "june",
            "july", "august", "september", "october", "november", "december"
        ]
        if any(m in q_lower for m in month_names):
            for opt in options:
                o = str(opt.get("option", "")).lower()
                if o in ["monthly", "month"]:
                    return opt

        # --- Quarter keywords → Quarterly ---
        quarter_keywords = ["quarter", "quarterly", "q1", "q2", "q3", "q4",
                            "jul-sep", "oct-dec", "jan-mar", "apr-jun"]
        if any(qk in q_lower for qk in quarter_keywords):
            for opt in options:
                if str(opt.get("option", "")).lower() in ["quarterly"]:
                    return opt

        # --- Year format "2023-24" or standalone year → Annually ---
        if re.search(r"\d{4}[-/]\d{2,4}", q_lower) or YEAR_PATTERN.search(q_lower):
            for opt in options:
                if str(opt.get("option", "")).lower() in ["annually", "annual"]:
                    return opt

        # --- No frequency clue → Select All ---
        return {
            "parent": options[0]["parent"],
            "filter_name": filter_name,
            "option": "Select All"
        }
    # =========================
    # YEAR FILTER (Year, financial_Year)
    # User mention nahi kiya → Select All (agar hai), else latest year
    # User mention kiya → exact year
    # =========================
    if "year" in fname_lower and "base" not in fname_lower:
        # Dataset intent ne agar specific year diya hai to pehle wahi try karo
        if isinstance(hint_value, int):
            mapped_from_hint = map_year_to_option(hint_value, options)
            if mapped_from_hint:
                return mapped_from_hint

        year_match = YEAR_PATTERN.search(q_lower)
        # If query is talking about base year (e.g. "base year 2010"), don't treat that as a data-year mention.
        if year_match:
            ytxt = year_match.group(1)
            if ("base year" in q_lower) or re.search(rf"\\bbase\\s*year\\s*{ytxt}\\b", q_lower) or re.search(rf"\\bbase\\s*{ytxt}\\b", q_lower):
                # region agent log
                _agent_log(
                    runId="pre-fix",
                    hypothesisId="Y",
                    location="running_code.py:select_best_filter_option:year",
                    message="Ignoring year match because it looks like base year mention",
                    data={"filter_name": filter_name, "matched_year": ytxt},
                )
                # endregion agent log
                year_match = None

        if not year_match:
            # User ne year nahi bola → Select All agar options mein hai, else latest year
            for opt in options:
                o = str(opt.get("option", "")).strip().lower()
                if o in ("select all", "selectall"):
                    return opt
            # Select All nahi hai → latest/current year return karo
            def _extract_year_val(o):
                m = re.search(r"20\d{2}", str(o.get("option", "")))
                return int(m.group(0)) if m else 0
            return max(options, key=lambda o: _extract_year_val(o))

        user_year = year_match.group(1)
        mapped = map_year_to_option(user_year, options)
        if mapped:
            return mapped

        # If exact mapping fails, pick closest available year option (deterministic).
        # Prefer Select All when present to maximize data availability.
        for opt in options:
            o = str(opt.get("option", "")).strip().lower()
            if o in ("select all", "selectall"):
                # region agent log
                _agent_log(
                    runId="pre-fix",
                    hypothesisId="Y",
                    location="running_code.py:select_best_filter_option:year",
                    message="Year not found in options; using Select All",
                    data={"filter_name": filter_name, "user_year": int(user_year)},
                )
                # endregion agent log
                return opt

        def _extract_year_val(o):
            m = re.search(r"20\d{2}", str(o.get("option", "")))
            return int(m.group(0)) if m else None

        user_y = int(user_year)
        scored = []
        for opt in options:
            oy = _extract_year_val(opt)
            if oy is None:
                continue
            scored.append((abs(oy - user_y), 0 if oy <= user_y else 1, oy, opt))
        if scored:
            scored.sort(key=lambda t: (t[0], t[1], t[2]))  # closest, prefer <= user year, then smaller
            best = scored[0][3]
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="Y",
                location="running_code.py:select_best_filter_option:year",
                message="Year not found in options; using closest available year",
                data={
                    "filter_name": filter_name,
                    "user_year": user_y,
                    "picked_option": best.get("option"),
                },
            )
            # endregion agent log
            return best

        pairs = [(query, f"{filter_name} {o['option']}") for o in options]
        scores = cross_encoder.predict(pairs)
        return options[int(np.argmax(scores))]

    # =========================
    # SERIES FILTER (CPI, NAS - Current/Back)
    # =========================
    if fname_lower == "series":
        # Dataset intent layer hint (e.g. CPI Back/Current, NAS Current/Constant)
        if isinstance(hint_value, str):
            hv = hint_value.lower()
            for opt in options:
                if str(opt.get("option", "")).lower() == hv:
                    return opt

        if "back" in q_lower or "historical" in q_lower:
            for opt in options:
                if str(opt.get("option", "")).lower() == "back":
                    return opt
        if "current" in q_lower:
            for opt in options:
                if str(opt.get("option", "")).lower() == "current":
                    return opt
        for opt in options:
            if str(opt.get("option", "")).lower() == "current":
                return opt
        return options[0] if options else {"parent": "", "filter_name": filter_name, "option": "Select All"}

    # =========================
    # CLASSIFICATION YEAR (ASI)
    # =========================
    if fname_lower == "classification_year":
        # Hint year (ASI classification_year) → direct map
        if isinstance(hint_value, int):
            mapped_from_hint = map_year_to_option(hint_value, options)
            if mapped_from_hint:
                return mapped_from_hint

        for opt in options:
            opt_text = str(opt.get("option", "")).lower()
            if opt_text in q_lower:
                return opt
        def _extract_year(opt):
            m = re.search(r"\d{4}", str(opt.get("option", "")))
            return int(m.group(0)) if m else 0
        return max(options, key=lambda o: _extract_year(o))

    # =========================
    # BASE YEAR FILTER (FINAL FIX)
    # =========================
    if "base" in fname_lower and "year" in fname_lower:

        # 🔹 hint provided by dataset intent layer (e.g. IIP base year, CPI base year)
        if isinstance(hint_value, str):
            hv = hint_value.lower()
            for opt in options:
                if hv in str(opt.get("option", "")).lower():
                    return opt

        # 🔹 check if user explicitly mentioned base year
        for opt in options:
            opt_text = str(opt.get("option", "")).lower()
            if opt_text in q_lower:
                return opt

        # 🔹 user ne base year nahi bola → latest base year pick karo
        def extract_start_year(opt):
            m = re.search(r"\d{4}", str(opt.get("option", "")))
            return int(m.group(0)) if m else 0

        latest = max(options, key=lambda o: extract_start_year(o))
        return latest

    # =========================
    # MONTH FILTER (WPI, IIP etc - calendar month: January, February, ...)
    # =========================
    if fname_lower == "month":
        month_map = [
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may", "may"), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep"), ("october", "oct"), ("november", "nov"), ("december", "dec")
        ]
        for full, short in month_map:
            if full in q_lower or short in q_lower:
                for opt in options:
                    if str(opt.get("option", "")).lower() == full or str(opt.get("option", "")).lower().startswith(full[:3]):
                        return opt
        return {
            "parent": options[0]["parent"],
            "filter_name": filter_name,
            "option": "Select All"
        }

    # =========================
    # OTHER FILTERS
    # =========================
    mentioned = []

    for opt in options:
        opt_text = str(opt.get("option", "")).lower().strip()
        if not opt_text:
            continue

        if opt_text in q_lower:
            mentioned.append(opt)
            continue

        for word in q_lower.split():
            if difflib.SequenceMatcher(None, opt_text, word).ratio() > 0.80:
                mentioned.append(opt)
                break

    if mentioned:
        pairs = [(query, f"{filter_name} {o['option']}") for o in mentioned]
        scores = cross_encoder.predict(pairs)
        return mentioned[int(np.argmax(scores))]

    return {
        "parent": options[0]["parent"],
        "filter_name": filter_name,
        "option": "Select All"
    }


# ================================
# LOAD PRODUCTS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_FILE = os.path.join(BASE_DIR, "products.json")
if not os.path.exists(PRODUCTS_FILE):
    PRODUCTS_FILE = os.path.join(BASE_DIR, "products", "products.json")
if not os.path.exists(PRODUCTS_FILE):
    raise FileNotFoundError(f"products.json not found. Tried: {BASE_DIR}/products.json and {BASE_DIR}/products/products.json")

with open(PRODUCTS_FILE, "r", encoding="utf-8", errors="ignore") as f:
    raw_products = json.load(f)

DATASETS, INDICATORS, FILTERS = [], [], []

for ds_name, ds_info in raw_products.get("datasets", {}).items():
    DATASETS.append({"code": ds_name, "name": ds_name})

    for ind in ds_info.get("indicators", []):
        ind_code = f"{ds_name}_{ind['name']}"
        INDICATORS.append({
            "code": ind_code,
            "name": ind["name"],
            "desc": ind.get("description", ""),
            "parent": ds_name
        })

        flat = universal_filter_normalizer(ind_code, ind.get("filters", []))
        FILTERS.extend(flat)

print(f"[INFO] DATASETS={len(DATASETS)}, INDICATORS={len(INDICATORS)}, FILTERS={len(FILTERS)}")

# ================================
# MODELS
# ================================
import threading

bi_encoder = None
cross_encoder = None
_SEARCH_INIT_READY = False
_SEARCH_INIT_ERROR = None
_SEARCH_INIT_LOCK = threading.Lock()
_RERANK_READY = False
_RERANK_ERROR = None
_RERANK_LOCK = threading.Lock()
ENABLE_VECTOR_INIT = False  # vector init hangs; keep BM25+rerank serving

# ================================
# VECTOR DB
# ================================
COLLECTION = "indicators_collection"

qclient = None
faiss_index = None
embeddings = None

if USE_QDRANT:
    try:
        qclient = QdrantClient(url="http://localhost:6333")
        if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
            qclient.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=qmodels.VectorParams(size=VECTOR_DIM,distance=qmodels.Distance.COSINE)
            )
        print("[INFO] Qdrant ready")
    except Exception as e:
        USE_QDRANT = False
        print("[WARN] Qdrant failed, using FAISS:", e)

def _init_reranker():
    """Init reranker only (cross-encoder). Much cheaper than embeddings."""
    global cross_encoder, _RERANK_READY, _RERANK_ERROR
    with _RERANK_LOCK:
        if _RERANK_READY or _RERANK_ERROR is not None:
            return
        try:
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_reranker:begin",
                message="Initializing cross-encoder reranker",
                data={},
            )
            # endregion agent log
            cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
            _RERANK_READY = True
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_reranker:ready",
                message="Cross-encoder ready",
                data={},
            )
            # endregion agent log
        except Exception as e:
            _RERANK_ERROR = str(e)
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_reranker:error",
                message="Cross-encoder init failed",
                data={"error": _RERANK_ERROR},
            )


def _ensure_reranker_ready():
    if _RERANK_READY:
        return True
    if _RERANK_ERROR is not None:
        return False
    threading.Thread(target=_init_reranker, daemon=True).start()
    return False

# Warm up reranker on startup to avoid first-request 503
threading.Thread(target=_init_reranker, daemon=True).start()


def _init_search_stack():
    """Heavy init (bi-encoder + embeddings + vector index). Runs in background."""
    global bi_encoder, qclient, faiss_index, embeddings, _SEARCH_INIT_READY, _SEARCH_INIT_ERROR, USE_QDRANT
    with _SEARCH_INIT_LOCK:
        if _SEARCH_INIT_READY or _SEARCH_INIT_ERROR is not None:
            return
        try:
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:begin",
                message="Initializing search stack (models + embeddings + index)",
                data={"indicators_n": len(INDICATORS), "use_qdrant": bool(USE_QDRANT)},
            )
            # endregion agent log
            # Force CPU: avoids occasional accelerator/GPU hangs on Windows during encode()
            bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")

            VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()

            if USE_QDRANT:
                try:
                    qclient = QdrantClient(url="http://localhost:6333")
                    if COLLECTION not in [c.name for c in qclient.get_collections().collections]:
                        qclient.recreate_collection(
                            collection_name=COLLECTION,
                            vectors_config=qmodels.VectorParams(size=VECTOR_DIM, distance=qmodels.Distance.COSINE),
                        )
                    print("[INFO] Qdrant ready")
                except Exception as e:
                    USE_QDRANT = False
                    qclient = None
                    print("[WARN] Qdrant failed, using FAISS:", e)

            names = [clean_text(i["name"]) for i in INDICATORS]
            descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

            def _encode_batched(texts, *, label: str, batch_size: int = 64):
                """Encode in small batches to avoid long single-call hangs."""
                vecs = []
                n = len(texts)
                i = 0
                while i < n:
                    j = min(i + batch_size, n)
                    batch = texts[i:j]
                    # region agent log
                    if i == 0:
                        _agent_log(
                            runId="pre-fix",
                            hypothesisId="S",
                            location="running_code.py:init_search_stack:encode_batch:begin",
                            message="Encoding first batch",
                            data={"label": label, "batch_size": len(batch)},
                        )
                    # endregion agent log
                    vecs.append(
                        bi_encoder.encode(
                            batch,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                            batch_size=batch_size,
                            device="cpu",
                        )
                    )
                    i = j
                    # region agent log
                    if i in (batch_size, n // 2, n):
                        _agent_log(
                            runId="pre-fix",
                            hypothesisId="S",
                            location="running_code.py:init_search_stack:encode_progress",
                            message="Batch encode progress",
                            data={"label": label, "done": i, "total": n},
                        )
                    # endregion agent log
                return np.vstack(vecs) if vecs else np.zeros((0, VECTOR_DIM), dtype="float32")

            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:before_embeddings",
                message="Starting embeddings computation",
                data={"names_n": len(names), "descs_n": len(descs)},
            )
            # endregion agent log

            # region agent log
            t0 = int(datetime.utcnow().timestamp() * 1000)
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:encode_names:begin",
                message="Encoding names started",
                data={"n": len(names)},
            )
            # endregion agent log
            names_vec = _encode_batched(names, label="names", batch_size=64)
            # region agent log
            t1 = int(datetime.utcnow().timestamp() * 1000)
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:encode_names:end",
                message="Encoding names finished",
                data={"ms": t1 - t0, "shape0": int(getattr(names_vec, "shape", [0])[0]) if names_vec is not None else 0},
            )
            # endregion agent log

            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:encode_descs:begin",
                message="Encoding descs started",
                data={"n": len(descs)},
            )
            # endregion agent log
            descs_vec = _encode_batched(descs, label="descs", batch_size=64)
            # region agent log
            t2 = int(datetime.utcnow().timestamp() * 1000)
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:encode_descs:end",
                message="Encoding descs finished",
                data={"ms": t2 - t1, "shape0": int(getattr(descs_vec, "shape", [0])[0]) if descs_vec is not None else 0},
            )
            # endregion agent log

            embeddings = (0.4 * names_vec + 0.6 * descs_vec)
            embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:after_embeddings",
                message="Finished embeddings computation",
                data={"vectors_n": int(getattr(embeddings, "shape", [0])[0]) if embeddings is not None else 0},
            )
            # endregion agent log

            if USE_QDRANT and qclient:
                qclient.upsert(
                    collection_name=COLLECTION,
                    points=[qmodels.PointStruct(id=i, vector=embeddings[i].tolist(), payload=INDICATORS[i]) for i in range(len(INDICATORS))],
                )
            else:
                faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
                faiss_index.add(embeddings.astype("float32"))

            _SEARCH_INIT_READY = True
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:ready",
                message="Search stack ready",
                data={"use_qdrant": bool(USE_QDRANT), "has_faiss": bool(faiss_index is not None)},
            )
            # endregion agent log
        except Exception as e:
            _SEARCH_INIT_ERROR = str(e)
            # region agent log
            _agent_log(
                runId="pre-fix",
                hypothesisId="S",
                location="running_code.py:init_search_stack:error",
                message="Search stack initialization failed",
                data={"error": _SEARCH_INIT_ERROR},
            )
            # endregion agent log


def _ensure_search_ready():
    if _SEARCH_INIT_READY:
        return True
    if _SEARCH_INIT_ERROR is not None:
        return False
    # kick background init (non-blocking)
    threading.Thread(target=_init_search_stack, daemon=True).start()
    return False

# ================================
# BM25 INDEX (Hybrid search)
# ================================
def _tokenize(text):
    """Simple tokenizer: lowercase, split on non-alnum."""
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip().split()

_bm25_corpus = [_tokenize(i["name"] + " " + i.get("desc", "")) for i in INDICATORS]
bm25_index = BM25Okapi(_bm25_corpus)
RRF_K = 60

def _rrf_fusion(vector_ranked_indices, bm25_ranked_indices, k=60):
    """Reciprocal Rank Fusion: score(d) = sum 1/(k+rank). Returns indices sorted by RRF score."""
    scores = {}
    for rank, idx in enumerate(vector_ranked_indices):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_ranked_indices):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

# ================================
# SEARCH
# ================================
def search_indicators(query, top_k=25, max_products=3, use_hybrid=True):
    """Hybrid (default) or vector-only search. Vector+BM25+RRF fusion, cross-encoder rerank, CPI conflict, max 1 per dataset."""
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="D",
        location="running_code.py:search_indicators:entry",
        message="search_indicators called",
        data={
            "use_hybrid": bool(use_hybrid),
            "top_k": int(top_k),
            "max_products": int(max_products),
            "vector_ready": bool(_SEARCH_INIT_READY and bi_encoder is not None and (qclient is not None or faiss_index is not None)),
            "reranker_ready": bool(_RERANK_READY and cross_encoder is not None),
        },
    )
    # endregion agent log
    # If vector index not ready, fall back to BM25-only candidate generation.
    if _SEARCH_INIT_READY and bi_encoder is not None and (qclient is not None or faiss_index is not None):
        q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
        q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)
        if USE_QDRANT and qclient:
            hits = qclient.search(collection_name=COLLECTION, query_vector=q_vec[0].tolist(), limit=top_k)
            vector_indices = [h.id for h in hits]
        else:
            _, I = faiss_index.search(q_vec.astype("float32"), top_k)
            vector_indices = [int(i) for i in I[0] if i >= 0]

        if use_hybrid:
            q_tokens = _tokenize(query)
            bm25_scores = bm25_index.get_scores(q_tokens)
            bm25_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()
            fused_indices = _rrf_fusion(vector_indices, bm25_indices, k=RRF_K)[:top_k]
        else:
            fused_indices = vector_indices
    else:
        q_tokens = _tokenize(query)
        bm25_scores = bm25_index.get_scores(q_tokens)
        fused_indices = np.argsort(bm25_scores)[::-1][:top_k].tolist()

    candidates = [INDICATORS[i] for i in fused_indices if 0 <= i < len(INDICATORS)]

    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="D",
        location="running_code.py:search_indicators:before_rerank",
        message="About to rerank candidates",
        data={"candidates_n": len(candidates)},
    )
    # endregion agent log
    scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="D",
        location="running_code.py:search_indicators:after_rerank",
        message="Rerank scores computed",
        data={"scores_n": int(getattr(scores, "shape", [len(scores)])[0]) if scores is not None else 0},
    )
    # endregion agent log
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # CPI conflict resolve ONLY if both present
    candidates = resolve_cpi_conflict(candidates, query)

    seen, final = set(), []
    for c in candidates:
        if c["parent"] not in seen:
            # Golden rule: CPI/CPI2 ke liye indicator-level keyword match se best choose karo
            if c["parent"] in ("CPI", "CPI2"):
                pool = [x for x in candidates if x["parent"] == c["parent"]]
                c = _choose_best_cpi_indicator(pool, query) or c
            seen.add(c["parent"])
            final.append(c)
        if len(final) == max_products:
            break

    # CPI vs CPIALRL vs WPI disambiguation: query intent ke hisaab se order fix (narrow scope)
    final = _reorder_final_by_cpi_cpialrl_wpi_intent(final, query)

    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="D",
        location="running_code.py:search_indicators:exit",
        message="search_indicators returning results",
        data={"final_n": len(final), "parents": [x.get("parent") for x in final]},
    )
    # endregion agent log
    return final


def _search_dataset_only(query, parent_codes):
    """Sirf given dataset(s) ke indicators mein search karo. Best matching indicator return, nahi mile to None."""
    if isinstance(parent_codes, str):
        parent_codes = (parent_codes,)
    indicators = [i.copy() for i in INDICATORS if i["parent"] in parent_codes]
    if not indicators:
        return None
    pairs = [(query, c["name"] + " " + c.get("desc", "")) for c in indicators]
    scores = cross_encoder.predict(pairs)
    for i, c in enumerate(indicators):
        c["score"] = float(scores[i])
    return max(indicators, key=lambda x: x["score"])


def _search_wpi_only(query):
    """WPI dataset ke andar sirf search karo. Force-include ke liye use hota hai."""
    return _search_dataset_only(query, "WPI")


def _search_ec_only(query):
    """EC4/EC5/EC6 ke andar search karo. Economic Census force-include ke liye."""
    return _search_dataset_only(query, ("EC4", "EC5", "EC6"))


###################query capture 


import uuid
from datetime import datetime

LOG_FILE = os.path.join(BASE_DIR, "logs", "queries.jsonl")

def save_query_log(raw_query, rewritten_query, response_json):
    """Har search request ko logs/queries.jsonl mein append karo (raw query, rewritten, response). Debug/analytics ke liye."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "raw_query": raw_query,
        "rewritten_query": rewritten_query,
        "response": response_json
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ================================
# DATASET-SPECIFIC INTENT → FILTER HINTS
# ================================

def compute_filter_hints(parent_code: str, raw_query: str, rewritten_query: str | None = None) -> dict:
    """
    Narrow, dataset-specific helper:
    - For a given dataset (parent_code) and query text, return hints for essential filters:
      e.g. {"Series": "Back", "Base_Year": "2010", "Frequency": "Quarterly"}.
    - Golden rule: dataset-scoped only; other datasets unaffected.
    """
    text = f"{raw_query or ''} {(rewritten_query or '')}".lower()
    hints: dict[str, str | int] = {}

    # -----------------------------
    # CPI / CPI2  (Base_Year + Series where applicable)
    # -----------------------------
    if parent_code == "CPI":
        # CPI has Base_Year: 2012/2010 and Series: Current/Back
        if "base year 2010" in text or "2010 base year" in text or re.search(r"\bbase\s*year\s*2010\b", text):
            hints["Base_Year"] = "2010"
            hints["Series"] = "Back"
        elif "base year 2012" in text or "2012 base year" in text or re.search(r"\bbase\s*year\s*2012\b", text):
            hints["Base_Year"] = "2012"
            hints["Series"] = "Current"
        elif "back series" in text or "back data" in text or "old series" in text or "historical" in text:
            # CPI back series defaults to 2010 base unless explicit says otherwise
            hints["Series"] = "Back"
            hints["Base_Year"] = "2010"
        elif "current series" in text or "latest series" in text or "new series" in text:
            hints["Series"] = "Current"
            hints["Base_Year"] = "2012"

    if parent_code == "CPI2":
        # CPI2 has Base_Year: 2024 and no Series filter in products.json
        hints["Base_Year"] = "2024"
        # If user explicitly wants combined/rural/urban, keep as hint for Sector (exact option labels exist)
        if "combined" in text:
            hints["Sector"] = "Combined"
        elif "urban" in text:
            hints["Sector"] = "Urban"
        elif "rural" in text:
            hints["Sector"] = "Rural"

    # -----------------------------
    # NAS (Series + Frequency)
    # -----------------------------
    if parent_code == "NAS":
        if "constant price" in text or "constant prices" in text:
            hints["Series"] = "Constant Price"
        elif "current price" in text or "current prices" in text:
            hints["Series"] = "Current Price"

        if "quarterly" in text or "q1" in text or "q2" in text or "q3" in text or "q4" in text:
            hints["Frequency"] = "Quarterly"
        elif "monthly" in text:
            hints["Frequency"] = "Monthly"
        elif "annual" in text or "annually" in text:
            hints["Frequency"] = "Annual"

    # -----------------------------
    # IIP (Base_Year / Financial_year)
    # -----------------------------
    if parent_code == "IIP":
        m = re.search(r"(20\d{2}[-/]\d{2,4})", text)
        if m and "base" in text:
            hints["Base_Year"] = m.group(1)
            hints["Financial_year"] = m.group(1)

    # -----------------------------
    # ASI (classification_year)
    # -----------------------------
    if parent_code == "ASI":
        m = YEAR_PATTERN.search(text)
        if m:
            hints["classification_year"] = int(m.group(1))

    # -----------------------------
    # PLFS (Year hint only; other filters already handled via synonym map)
    # -----------------------------
    if parent_code == "PLFS":
        m = YEAR_PATTERN.search(text)
        if m:
            hints["Year"] = int(m.group(1))

    # -----------------------------
    # ENVSTAT / TUS / Gender / HCES / ESI:
    # Most filter semantics (Gender/Sector/State) already covered by rewrite rules
    # and generic smart filter engine; we only steer Frequency when obvious.
    # -----------------------------
    if parent_code in ("ENVSTAT", "TUS", "Gender", "HCES", "ESI"):
        if "quarterly" in text:
            hints["Frequency"] = "Quarterly"
        elif "monthly" in text:
            hints["Frequency"] = "Monthly"
        elif "annual" in text or "annually" in text:
            hints["Frequency"] = "Annual"

    return hints


# ================================
# FLASK
# ================================
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    """Home page - search UI render karo."""
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="Z",
        location="running_code.py:home",
        message="Home route hit",
        data={},
    )
    # endregion agent log
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    """Health/readiness probe for cold start."""
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="H",
        location="running_code.py:health",
        message="Health route hit",
        data={"reranker_ready": bool(_RERANK_READY), "vector_ready": bool(_SEARCH_INIT_READY)},
    )
    # endregion agent log
    status = 200 if _RERANK_READY else 503
    return jsonify(
        {
            "ok": True,
            "rerankerReady": bool(_RERANK_READY),
            "vectorReady": bool(_SEARCH_INIT_READY),
            "vectorInitEnabled": bool(ENABLE_VECTOR_INIT),
        }
    ), status

@app.route("/search/predict", methods=["POST"])
def predict():
    """Main API: query receive karo, LLM rewrite, semantic search, filter selection, results + filters return. Top 3 datasets."""
    raw_q = request.json.get("query", "").strip()
    if not raw_q:
        return jsonify({"error": "query required"}), 400
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="A",
        location="running_code.py:predict:entry",
        message="Received search request",
        data={"raw_q_len": len(raw_q), "raw_q_snippet": raw_q[:160]},
    )
    # endregion agent log

    #  LLM rewrite
    q = rewrite_query_with_llm(raw_q)

    # Ensure reranker for search + filter option selection
    _agent_log(
        runId="pre-fix",
        hypothesisId="S",
        location="running_code.py:predict:ensure_reranker_ready",
        message="Ensuring reranker ready",
        data={"ready": bool(_RERANK_READY), "has_error": bool(_RERANK_ERROR is not None)},
    )
    if not _ensure_reranker_ready():
        # Try a synchronous init so first request doesn't 503.
        if _RERANK_ERROR is None:
            _init_reranker()
        if _RERANK_ERROR is not None or not _RERANK_READY:
            return jsonify({"error": "reranker initialization failed", "details": _RERANK_ERROR or "unknown"}), 500

    # Vector stack init currently hangs in bi-encoder encode; keep disabled by default.
    if ENABLE_VECTOR_INIT:
        _agent_log(
            runId="pre-fix",
            hypothesisId="S",
            location="running_code.py:predict:ensure_search_ready",
            message="Ensuring vector stack ready (best-effort)",
            data={"ready": bool(_SEARCH_INIT_READY), "has_error": bool(_SEARCH_INIT_ERROR is not None)},
        )
        _ensure_search_ready()
    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="A",
        location="running_code.py:predict:after_rewrite",
        message="Rewritten query produced",
        data={"rewritten_len": len(q or ""), "rewritten_snippet": (q or "")[:200]},
    )
    # endregion agent log

    # Fallback: expand dataset short names for better semantic search
    q_lower = q.lower()
    if re.search(r'\bec\b', q_lower) and not any(x in q_lower for x in ["economic census", "ec4", "ec5", "ec6"]):
        q = q + " Economic Census"
    if re.search(r'\bwpi\b', q_lower) and not any(x in q_lower for x in ["wholesale price", "wholesale price index"]):
        q = q + " Wholesale Price Index"
    if re.search(r'\baishe\b', q_lower) and "higher education" not in q_lower:
        q = q + " All India Survey on Higher Education"
    if re.search(r'\bnfhs\b', q_lower) and "family health" not in q_lower:
        q = q + " National Family Health Survey"
    if re.search(r'\bnss', q_lower) and "national sample" not in q_lower:
        q = q + " National Sample Survey"

    print("RAW :", raw_q)
    print("LLM :", q)

    top_results = search_indicators(q)

    # Force-include EC: if user asked for EC (or "economic census") but no EC4/EC5/EC6 in results, add best EC match
    _ec_like = ("economic" in raw_q.lower() and "census" in raw_q.lower()) or bool(re.search(r'\bec\b', raw_q.lower()))
    ec_wanted = _ec_like and not any(x in raw_q.lower() for x in ["ec4", "ec5", "ec6"])
    ec_in_results = any(r["parent"] in ("EC4", "EC5", "EC6") for r in top_results)
    if ec_wanted and not ec_in_results:
        ec_best = _search_ec_only(q or raw_q)
        if ec_best:
            top_results = [ec_best] + [r for r in top_results if r["parent"] != ec_best["parent"]][:2]
            ec_best["score"] = max(r["score"] for r in top_results) + 1  # 95% confidence

    # Force-include WPI: if user asked for WPI (or "wholesale price") but no WPI in results, add best WPI match
    _wpi_like = re.search(r'\bwpi\b', raw_q.lower()) or ("wholesale" in raw_q.lower() and "price" in raw_q.lower())
    wpi_wanted = _wpi_like
    wpi_in_results = any(r["parent"] == "WPI" for r in top_results)
    if wpi_wanted and not wpi_in_results:
        wpi_best = _search_wpi_only(q or raw_q)
        if wpi_best:
            top_results = [wpi_best] + [r for r in top_results if r["parent"] != wpi_best["parent"]][:2]
            wpi_best["score"] = max(r["score"] for r in top_results) + 1  # 95% confidence

    # Force-include when user searched by dataset name but it's not in results
    _raw_lower = raw_q.lower().strip()
    _force_ds = None
    if re.search(r'\bnss77\b', _raw_lower):
        _force_ds = ["NSS77"]
    elif re.search(r'\bnss78\b', _raw_lower):
        _force_ds = ["NSS78"]
    elif re.search(r'\bnss79\b', _raw_lower) or re.search(r'\bnss79c\b', _raw_lower):
        _force_ds = ["NSS79C"]
    elif re.search(r'\bnss\b', _raw_lower):
        _force_ds = ["NSS77", "NSS78", "NSS79C"]
    elif re.search(r'\bnfhs\b', _raw_lower):
        _force_ds = ["NFHS"]
    elif re.search(r'\baishe\b', _raw_lower):
        _force_ds = ["AISHE"]
    elif re.search(r'\bcpi\b', _raw_lower) or ("inflation" in _raw_lower and "wholesale" not in _raw_lower and "wpi" not in _raw_lower):
        _force_ds = ["CPI", "CPI2"]
    elif re.search(r'\bplfs\b', _raw_lower):
        _force_ds = ["PLFS"]
    if _force_ds and not any(r["parent"] in _force_ds for r in top_results):
        ds_best = _search_dataset_only(q or raw_q, _force_ds)
        if ds_best:
            top_results = [ds_best] + [r for r in top_results if r["parent"] != ds_best["parent"]][:2]
            ds_best["score"] = max(r["score"] for r in top_results) + 1  # 95% confidence

    # Prioritize dataset to 1st when user searched by dataset name (or very strong intent words) - all 23 datasets, 95% confidence
    _raw_lower = raw_q.lower().strip()
    _ds_priority = None
    # Specific codes first (nss77 before nss, ec4 before ec, etc.)
    if re.search(r'\bnss77\b', _raw_lower):
        _ds_priority = ["NSS77"]
    elif re.search(r'\bnss78\b', _raw_lower):
        _ds_priority = ["NSS78"]
    elif re.search(r'\bnss79\b', _raw_lower) or re.search(r'\bnss79c\b', _raw_lower):
        _ds_priority = ["NSS79C"]
    elif re.search(r'\bec4\b', _raw_lower):
        _ds_priority = ["EC4"]
    elif re.search(r'\bec5\b', _raw_lower):
        _ds_priority = ["EC5"]
    elif re.search(r'\bec6\b', _raw_lower):
        _ds_priority = ["EC6"]
    elif re.search(r'\bcpi2\b', _raw_lower):
        _ds_priority = ["CPI2"]
    elif re.search(r'\bwpi\b', _raw_lower) or ("wholesale" in _raw_lower and "price" in _raw_lower):
        _ds_priority = ["WPI"]
    elif re.search(r'\bplfs\b', _raw_lower):
        _ds_priority = ["PLFS"]
    elif re.search(r'\bec\b', _raw_lower) or ("economic" in _raw_lower and "census" in _raw_lower):
        _ds_priority = ["EC4", "EC5", "EC6"]
    elif re.search(r'\bnss\b', _raw_lower):
        _ds_priority = ["NSS77", "NSS78", "NSS79C"]
    elif re.search(r'\bcpi\b', _raw_lower) or ("inflation" in _raw_lower and "wholesale" not in _raw_lower and "wpi" not in _raw_lower):
        _ds_priority = ["CPI", "CPI2"]
    elif re.search(r'\bcpialrl\b', _raw_lower) or ("consumer price" in _raw_lower and "agricultural" in _raw_lower):
        _ds_priority = ["CPIALRL"]
    elif re.search(r'\bnas\b', _raw_lower):
        _ds_priority = ["NAS"]
    elif re.search(r'\basi\b', _raw_lower):
        _ds_priority = ["ASI"]
    elif re.search(r'\bhces\b', _raw_lower):
        _ds_priority = ["HCES"]
    elif re.search(r'\biip\b', _raw_lower):
        _ds_priority = ["IIP"]
    elif re.search(r'\brbi\b', _raw_lower):
        _ds_priority = ["RBI"]
    elif re.search(r'\baishe\b', _raw_lower) or ("higher education" in _raw_lower and "survey" in _raw_lower):
        _ds_priority = ["AISHE"]
    elif re.search(r'\bnfhs\b', _raw_lower) or ("family health" in _raw_lower and "survey" in _raw_lower):
        _ds_priority = ["NFHS"]
    elif re.search(r'\btus\b', _raw_lower) or ("time use" in _raw_lower and "survey" in _raw_lower):
        _ds_priority = ["TUS"]
    elif re.search(r'\besi\b', _raw_lower) or ("employment" in _raw_lower and "survey" in _raw_lower and "establishment" in _raw_lower):
        _ds_priority = ["ESI"]
    elif re.search(r'\benvstat\b', _raw_lower) or ("environment" in _raw_lower and "statistic" in _raw_lower):
        _ds_priority = ["ENVSTAT"]
    elif re.search(r'\basuse\b', _raw_lower):
        _ds_priority = ["ASUSE"]
    if _ds_priority:
        for i, r in enumerate(top_results):
            if r["parent"] in _ds_priority:
                if i > 0:
                    top_results = [r] + [x for x in top_results if x["parent"] != r["parent"]][:2]
                    r = top_results[0]
                # Boost 95% confidence (whether moved or already 1st)
                all_scores = [x["score"] for x in top_results]
                top_results[0]["score"] = max(all_scores) + 1
                break

    # region agent log
    _agent_log(
        runId="pre-fix",
        hypothesisId="A",
        location="running_code.py:predict:after_top_results",
        message="Top results finalized (after force include + prioritize)",
        data={
            "top_parents": [r.get("parent") for r in (top_results or [])],
            "top_codes": [r.get("code") for r in (top_results or [])],
            "top_scores": [r.get("score") for r in (top_results or [])],
        },
    )
    # endregion agent log

    confidences = normalize_confidence([r["score"] for r in top_results])

    results = []

    for ind, conf in zip(top_results, confidences):
        dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
        related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]
        # region agent log
        _agent_log(
            runId="pre-fix",
            hypothesisId="A",
            location="running_code.py:predict:per_indicator_filters",
            message="Preparing grouped filters for indicator",
            data={
                "dataset_parent": ind.get("parent"),
                "indicator_code": ind.get("code"),
                "related_filters_n": len(related_filters),
            },
        )
        # endregion agent log

        grouped = {}
        for f in related_filters:
            grouped.setdefault(f["filter_name"], []).append(f)

        # Dataset-specific hints for this indicator
        hints = compute_filter_hints(ind["parent"], raw_q, q)

        best_filters = []
        for fname, opts in grouped.items():
            best_opt = select_best_filter_option(
                query=q,
                filter_name=fname,
                options=opts,
                cross_encoder=cross_encoder,
                hints=hints,
            )
            best_filters.append({
                "filter_name": fname,
                "option": best_opt["option"]
            })
        # Filter Accuracy (4) + Essential (CPI/IIP/ASI/NAS/CPIALRL): ensure present & order
        best_filters = ensure_required_filters_present(best_filters, ind["parent"], grouped, q, cross_encoder, hints=hints)

        results.append({
            "dataset": dataset["name"],
            "product": dataset["code"].lower(),  # ec4, ec5, ec6 - for URL (macroindicators?product=ec4)
            "indicator": ind["name"],
            "confidence": conf,
            "filters": best_filters
        })
    response = {"results": results}
        #  SAVE OUTPUT
    save_query_log(
        raw_query=raw_q,
        rewritten_query=q,
        response_json=response
    )

    #return jsonify(response)

    # region agent log
    try:
        _top = results[0] if results else None
        _top_filters = {f.get("filter_name"): f.get("option") for f in (_top.get("filters", []) if _top else [])}
        _agent_log(
            runId="pre-fix",
            hypothesisId="R",
            location="running_code.py:predict:response_summary",
            message="Returning response summary",
            data={
                "results_n": len(results),
                "top_dataset": _top.get("dataset") if _top else None,
                "top_indicator": _top.get("indicator") if _top else None,
                "top_confidence": _top.get("confidence") if _top else None,
                "top_filters_subset": {k: _top_filters.get(k) for k in ["Year", "Sector", "Gender", "State", "Series", "Base_Year"] if k in _top_filters},
            },
        )
    except Exception:
        pass
    # endregion agent log

    return jsonify({"results": results})

if __name__ == "__main__":
    # Disable auto-reloader to avoid double initialization on cold start.
    app.run(debug=True, host="0.0.0.0", port=5009, use_reloader=False)

