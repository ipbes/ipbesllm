#!/bin/sh
# Converts all 14 assessments, refreshes data/ttl, cleans the person data, then
# writes the comparisons and PDF reports. It does not index. Run: src/refresh_all.sh
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=src
A="AFA AMA APA BBA ECA GA1 IAS LDR NXS PPA SCM SUA TCA VA"
mkdir -p .cache/links
echo $A | tr ' ' '\n' | xargs -P 4 -I{} sh -c '.venv/bin/python src/pdf_to_ttl.py {} > .cache/links/{}.out 2>&1; echo {} exit $?' || exit 1
for a in $A; do cp data/pdf_compare/$a/${a}_pdf.ttl data/ttl/${a}.ttl; done
.venv/bin/python src/clean_person_data.py --apply > .cache/links/clean.out 2>&1 || exit 1
.venv/bin/python src/conversion_report.py --all > .cache/links/reports.out 2>&1
echo reports exit $?
