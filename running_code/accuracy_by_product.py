"""Product-by-product Dataset, Indicator, Filters, Essential Filters accuracy (no Excel needed)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing running_code...", flush=True)
import running_code

# Ensure reranker (cross-encoder) is ready for filter selection
running_code._ensure_reranker_ready()

MANDATORY_4 = running_code.MANDATORY_4
ESSENTIAL_FILTERS_BY_DATASET = running_code.ESSENTIAL_FILTERS_BY_DATASET


def _get_indicator_filter_names(ind_code):
    """Filter names available for this indicator."""
    return set(f["filter_name"] for f in running_code.FILTERS if f["parent"] == ind_code)


def run():
    indicators = running_code.INDICATORS
    by_product = {}
    for ind in indicators:
        p = ind.get("parent", "")
        if p not in by_product:
            by_product[p] = []
        by_product[p].append(ind)

    print("\n=== PRODUCT-BY-PRODUCT ACCURACY (Dataset, Indicator, Filters, Essential) ===\n")
    print(f"{'Product':<12} {'n':>4}  {'Dataset%':>8}  {'Indicator%':>10}  {'Filters%':>9}  {'Essential%':>10}")
    print("-" * 65)

    total_n = total_ds_ok = total_ind_ok = total_filt_ok = total_ess_ok = 0
    SAMPLE = 5 if "--quick" in sys.argv else 15
    results_by_product = []
    for product in sorted(by_product.keys()):
        inds = by_product[product]
        if len(inds) > SAMPLE:
            import random
            random.seed(42)
            inds = random.sample(inds, SAMPLE)
        n = len(inds)
        ds_ok = ind_ok = filt_ok = ess_ok = 0
        filt_denom = ess_denom = 0
        for ind in inds:
            name = ind["name"]
            full_results = running_code.get_full_results_with_filters(name, max_products=3)
            if not full_results:
                continue
            top = full_results[0]
            got_parent = (top.get("parent") or "").upper()
            exp_parent = product.upper()
            if got_parent == exp_parent:
                ds_ok += 1
            got_ind = (top.get("indicator") or top.get("name") or "").strip()
            exp_ind = str(name).strip()
            if got_ind == exp_ind or exp_ind[:50] in got_ind or got_ind[:50] in exp_ind:
                ind_ok += 1

            filters_list = top.get("filters") or []
            resp_filter_names = set(f.get("filter_name") for f in filters_list if f.get("filter_name"))
            ind_code = None
            for i in running_code.INDICATORS:
                if (i.get("parent") or "").upper() == (got_parent or "") and (
                    (i.get("name") or "").strip() == (top.get("indicator") or "").strip()
                    or (top.get("indicator") or "")[:50] in (i.get("name") or "")
                    or (i.get("name") or "")[:50] in (top.get("indicator") or "")
                ):
                    ind_code = i.get("code", "")
                    break
            if not ind_code:
                ind_code = f"{got_parent}_{top.get('indicator', '')}"
            indicator_filter_names = _get_indicator_filter_names(ind_code)

            mand_applicable = [m for m in MANDATORY_4 if m in indicator_filter_names]
            if mand_applicable:
                filt_denom += 1
                if all(m in resp_filter_names for m in mand_applicable):
                    filt_ok += 1
            ess_list = list(ESSENTIAL_FILTERS_BY_DATASET.get(got_parent, []))
            if got_parent in ("IIP", "NAS") and "Base_Year" in ess_list:
                q_lower = (name or "").lower()
                if "base" not in q_lower and "2010" not in q_lower and "2012" not in q_lower:
                    ess_list = [e for e in ess_list if e != "Base_Year"]
            ess_applicable = [e for e in ess_list if e in indicator_filter_names]
            if ess_applicable:
                ess_denom += 1
                if all(e in resp_filter_names for e in ess_applicable):
                    ess_ok += 1

        ds_pct = 100 * ds_ok / n if n else 0
        ind_pct = 100 * ind_ok / n if n else 0
        filt_pct = 100 * filt_ok / filt_denom if filt_denom else 100.0
        ess_pct = 100 * ess_ok / ess_denom if ess_denom else 100.0
        total_n += n
        total_ds_ok += ds_ok
        total_ind_ok += ind_ok
        total_filt_ok += filt_ok
        total_ess_ok += ess_ok
        results_by_product.append({
            "product": product, "n": n,
            "ds_ok": ds_ok, "ind_ok": ind_ok, "filt_ok": filt_ok, "ess_ok": ess_ok,
            "ds_pct": ds_pct, "ind_pct": ind_pct, "filt_pct": filt_pct, "ess_pct": ess_pct,
            "filt_denom": filt_denom, "ess_denom": ess_denom,
        })
        print(f"{product:<12} {n:>4}  {ds_pct:>7.1f}%  {ind_pct:>9.1f}%  {filt_pct:>8.1f}%  {ess_pct:>9.1f}%")

    print("-" * 65)
    total_ds_pct = 100 * total_ds_ok / total_n if total_n else 0
    total_ind_pct = 100 * total_ind_ok / total_n if total_n else 0
    total_filt_denom = sum(r["filt_denom"] for r in results_by_product)
    total_ess_denom = sum(r["ess_denom"] for r in results_by_product)
    total_filt_pct = 100 * total_filt_ok / total_filt_denom if total_filt_denom else 100.0
    total_ess_pct = 100 * total_ess_ok / total_ess_denom if total_ess_denom else 100.0
    if total_n:
        print(f"{'TOTAL':<12} {total_n:>4}  {total_ds_pct:>7.1f}%  {total_ind_pct:>9.1f}%  {total_filt_pct:>8.1f}%  {total_ess_pct:>9.1f}%")

    out_path = os.path.join(os.path.dirname(__file__), "accuracy_by_product_result.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("PRODUCT-BY-PRODUCT ACCURACY (Dataset %, Indicator %, Filters %, Essential %)\n")
        f.write("=" * 75 + "\n")
        for r in results_by_product:
            f.write(f"{r['product']:<12} n={r['n']:>4}  Dataset: {r['ds_pct']:>6.1f}%  Indicator: {r['ind_pct']:>6.1f}%  Filters: {r['filt_pct']:>6.1f}%  Essential: {r['ess_pct']:>6.1f}%\n")
        f.write("-" * 75 + "\n")
        f.write(f"{'TOTAL':<12} n={total_n:>4}  Dataset: {total_ds_pct:>6.1f}%  Indicator: {total_ind_pct:>6.1f}%  Filters: {total_filt_pct:>6.1f}%  Essential: {total_ess_pct:>6.1f}%\n")
    print(f"\nFull report saved to {out_path}")


if __name__ == "__main__":
    run()
