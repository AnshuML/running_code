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
from datetime import datetime
import difflib

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


def ensure_required_filters_present(best_filters, parent_code, grouped, query, cross_encoder):
    """Jo required filters (4 + essential) grouped mein hain lekin best_filters mein nahi, unko add karo. Phir sahi order apply karo."""
    required = list(MANDATORY_4)
    required.extend(ESSENTIAL_FILTERS_BY_DATASET.get(parent_code, []))
    required = list(dict.fromkeys(required))
    out_names = {f["filter_name"]: f for f in best_filters}
    for r in required:
        if r in out_names:
            continue
        if r not in grouped:
            continue
        opts = grouped[r]
        if not opts:
            continue
        best_opt = select_best_filter_option(query, r, opts, cross_encoder)
        best_filters.append({"filter_name": r, "option": best_opt["option"]})
        out_names[r] = best_filters[-1]
    best_filters = ensure_mandatory_filter_order(best_filters, parent_code)
    # Golden rule: CPI-only. Series+Base_Year valid combos: (Current,2012), (Back,2010)
    best_filters = ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query)
    return best_filters


def ensure_cpi_series_base_year_consistent(best_filters, parent_code, grouped, query):
    """CPI/CPI2 ke liye Series+Base_Year valid combo ensure karo: Current↔2012, Back↔2010. Mismatch ho to fix. Baaki datasets pe no-op."""
    if parent_code not in ("CPI", "CPI2"):
        return best_filters
    by_name = {f["filter_name"]: f for f in best_filters}
    if "Series" not in by_name or "Base_Year" not in by_name:
        return best_filters
    q_lower = query.lower()
    series_opt = str(by_name["Series"].get("option", "")).lower()
    base_opt = str(by_name["Base_Year"].get("option", "")).lower()
    base_year = re.search(r"20\d{2}", base_opt)
    base_year = base_year.group(0) if base_year else ""
    target_series, target_base = None, None
    if "back" in q_lower or "2010" in q_lower:
        target_series, target_base = "Back", "2010"
    elif "current" in q_lower or "2012" in q_lower:
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
def select_best_filter_option(query, filter_name, options, cross_encoder):
    """Query ke hisaab se sabse sahi filter option pick karo. Year→Select All/latest, Series→Current/Back, State/Sector→match, etc."""
    if not options:
        return {"parent": "", "filter_name": filter_name, "option": "Select All"}
    q_lower = query.lower()
    fname_lower = filter_name.lower()
     
    # =========================
    # FREQUENCY FILTER
    # =========================
    if fname_lower in ["frequency"]:
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
        year_match = YEAR_PATTERN.search(q_lower)

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

        pairs = [(query, f"{filter_name} {o['option']}") for o in options]
        scores = cross_encoder.predict(pairs)
        return options[int(np.argmax(scores))]

    # =========================
    # SERIES FILTER (CPI, NAS - Current/Back)
    # =========================
    if fname_lower == "series":
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
bi_encoder = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# ================================
# VECTOR DB
# ================================
VECTOR_DIM = bi_encoder.get_sentence_embedding_dimension()
COLLECTION = "indicators_collection"

qclient = None
faiss_index = None

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

names = [clean_text(i["name"]) for i in INDICATORS]
descs = [clean_text(i.get("desc", "")) for i in INDICATORS]

embeddings = (0.4 * bi_encoder.encode(names, convert_to_numpy=True) + 0.6 * bi_encoder.encode(descs, convert_to_numpy=True))
embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

if USE_QDRANT and qclient:
    qclient.upsert(
        collection_name=COLLECTION,
        points=[qmodels.PointStruct(id=i,vector=embeddings[i].tolist(),payload=INDICATORS[i]) for i in range(len(INDICATORS))]
    )
else:
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings.astype("float32"))

# ================================
# SEARCH
# ================================
def search_indicators(query, top_k=25, max_products=3):
    """Query ke hisaab se semantic search karo (Qdrant/FAISS). Cross-encoder rerank, CPI conflict resolve. Har dataset se max 1 indicator. Top max_products return."""
    q_vec = bi_encoder.encode([clean_text(query)], convert_to_numpy=True)
    q_vec /= np.linalg.norm(q_vec, axis=1, keepdims=True)

    if USE_QDRANT and qclient:
        hits = qclient.search(collection_name=COLLECTION,query_vector=q_vec[0].tolist(),limit=top_k)
        candidates = [h.payload for h in hits]
    else:
        _, I = faiss_index.search(q_vec.astype("float32"), top_k)
        candidates = [INDICATORS[i] for i in I[0] if i >= 0]

    scores = cross_encoder.predict([(query, c["name"] + " " + c.get("desc", "")) for c in candidates])
    for i, c in enumerate(candidates):
        c["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # CPI conflict resolve ONLY if both present
    candidates = resolve_cpi_conflict(candidates, query)

    seen, final = set(), []
    for c in candidates:

        if c["parent"] not in seen:
            seen.add(c["parent"])
            final.append(c)
        if len(final) == max_products:
            break


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
# FLASK
# ================================
app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def home():
    """Home page - search UI render karo."""
    return render_template("index.html")

@app.route("/search/predict", methods=["POST"])
def predict():
    """Main API: query receive karo, LLM rewrite, semantic search, filter selection, results + filters return. Top 3 datasets."""
    raw_q = request.json.get("query", "").strip()
    if not raw_q:
        return jsonify({"error": "query required"}), 400

    #  LLM rewrite
    q = rewrite_query_with_llm(raw_q)

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
    elif re.search(r'\bcpi\b', _raw_lower):
        _force_ds = ["CPI", "CPI2"]
    elif re.search(r'\bplfs\b', _raw_lower):
        _force_ds = ["PLFS"]
    if _force_ds and not any(r["parent"] in _force_ds for r in top_results):
        ds_best = _search_dataset_only(q or raw_q, _force_ds)
        if ds_best:
            top_results = [ds_best] + [r for r in top_results if r["parent"] != ds_best["parent"]][:2]
            ds_best["score"] = max(r["score"] for r in top_results) + 1  # 95% confidence

    # Prioritize dataset to 1st when user searched by dataset name - all 23 datasets, 95% confidence
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
    elif re.search(r'\bcpi\b', _raw_lower):
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

    confidences = normalize_confidence([r["score"] for r in top_results])

    results = []

    for ind, conf in zip(top_results, confidences):
        dataset = next(d for d in DATASETS if d["code"] == ind["parent"])
        related_filters = [f for f in FILTERS if f["parent"] == ind["code"]]

        grouped = {}
        for f in related_filters:
            grouped.setdefault(f["filter_name"], []).append(f)

        best_filters = []
        for fname, opts in grouped.items():
            best_opt = select_best_filter_option(
                query=q,
                filter_name=fname,
                options=opts,
                cross_encoder=cross_encoder
            )
            best_filters.append({
                "filter_name": fname,
                "option": best_opt["option"]
            })
        # Filter Accuracy (4) + Essential (CPI/IIP/ASI/NAS/CPIALRL): ensure present & order
        best_filters = ensure_required_filters_present(best_filters, ind["parent"], grouped, q, cross_encoder)

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

    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5009)

