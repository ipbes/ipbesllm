#!/usr/bin/env python3
"""Clean person and chapter links in the assessment .ttl files.

Safe to run repeatedly: on data that is already clean it changes nothing. Re-run it
after re-exporting any .ttl file, then reload the data (see README, "Cleaning person
data"). It only edits files under ipbes-resource/files/, never Fuseki.

    python3 ipbes-scripts/clean-person-data.py              # dry run, writes nothing
    python3 ipbes-scripts/clean-person-data.py --verbose    # also list every merge
    python3 ipbes-scripts/clean-person-data.py --apply      # write the changes

It does three things:

1. Broken person URIs. A chapter's foaf:Person list may name a person URI that has no
   record (Babai_Dniel for Babai_Dániel). It is repointed only if exactly one person
   in the same file has a matching name AND lists that chapter themselves.
2. Dangling chapter links. A person's ipbes:Chapter may name a chapter with no record.
   CHAPTER_REPOINT maps those to the real chapter; CHAPTER_DROP removes values that
   are not chapters at all, such as report-level co-chair roles.
3. Duplicate people. Records with the same name (ignoring case, accents, punctuation
   and word order) are merged, plus the reviewed lists SAME_PERSON and SAME_PERSON_INITIALS.
   One person, one ORCID: records with different ORCIDs are never merged. A shared
   ORCID alone is never a reason to merge either, because the data has shared ORCIDs
   that are mistakes. Each retired URI keeps `dcterms:isReplacedBy`, which the
   resolver follows, so published URIs keep working.

Only standard-library Python is needed.
"""
import argparse
import collections
import difflib
import pathlib
import re
import sys
import unicodedata

NS = 'http://ontology.ipbes.net/report/person/'
CH = 'http://ontology.ipbes.net/report/ch/'
DEFAULT_DIR = pathlib.Path(__file__).resolve().parent.parent / 'data' / 'ttl'
BOM = '﻿'

# ---------------------------------------------------------------------------------
# Reviewed decisions. Add to these lists rather than editing the .ttl files by hand.
# ---------------------------------------------------------------------------------

# Step 1: broken URIs whose name is too damaged to match automatically.
BROKEN_URI_TARGETS = {
    'dosSantos_MariaJoãoFerreira': 'FerreiradosSantos_João_Maria',
    'dosSantos_MariaJooFerreira': 'FerreiradosSantos_João_Maria',
    'HernandezMrquez_GuadalupeYesenia': 'HernandezMárquez_Yesenia_Guadalupe',
    'Kouame_AmenanAgns': 'Kouame_Agnès_Amenan',
    'Payyappalli\xa0_UnniPayyappalli\xa0': 'Payyappallimana_Unnikrishnan',
    'Estoque_RonaldC.': 'Estoque_Canero_Ronald',
}
# Template rows from the source spreadsheets, not people.
PLACEHOLDER_REFERENCES = {'Surname_Name'}
PLACEHOLDER_RECORDS = {'Lastname_M._Firstname'}

# Step 2: chapter URIs (the part after /report/ch/).
CHAPTER_REPOINT = {f'ECA18chCh{i}': f'ECA18ch{i}' for i in range(1, 7)}
CHAPTER_REPOINT.update({f'GA119ch2.{i}': f'GA119ch2-{i}' for i in range(1, 4)})
CHAPTER_REPOINT[f'VA22ch{BOM}1'] = 'VA22ch1'
CHAPTER_DROP = {'ECA18ch', 'SCM16', 'TCA24ch', 'TCA24chall', 'SUA22chSPM', 'LDR18chN/A'}

# Step 3: same ORCID and compatible names, but the spelling differs too much to match.
# Do not add a pair on ORCID alone: check that the names agree.
SAME_PERSON = [
    ['SonnewendBrondízio_Eduardo', 'brondizio_eduardo', 'brondízio_eduardo'],
    ['Shrestha_UttamBabu', 'shrestha_uttam'],
    ['Dolors_Armenteras_Pascual', 'armenteras_dolors'],
    ['HenriqueSantinBrancalion_Pedro', 'brancalion_pedro'],
    ['Wai_Lung_(William)_Cheung', 'William_Cheung', 'cheung_william'],
    ['ReyesGarca_Victoria', 'garcía_victoria', 'Garcia_Reyes_Victoria'],
    ['Gautam_Prasad_Ambika', 'gautam_ambika'],
    ['Gould_K._Rachelle', 'Gould_RachelleK.', 'gould_rachelle'],
    ['hendr_andrew', 'hendry_andrew'],
    ['Kok_T.J._Marcel', 'kok_marcel'],
    ['Krug_M._Rainer', 'krug_rainer', 'Krug_Rainer'],
    ['Lynch_J._Abigail', 'lynch_abigail'],
    ['Horcea-Milcu_Andra-Ioana', 'milcu_andra'],
    ['Mwampamba_TuyeniH.', 'mwampamba_tuyeni'],
    ['Maria_Gabriela_Palomo', 'palomo_maria'],
    ['Subramanian_SuneethaM.', 'subramanian_suneetha'],
    ['FernandezTrigoso_Miguel', 'Fernández_Miguel'],
    ['Fisher_Judith', 'Fisher_L._Judith'],
    ['Rono_Betty', 'Rono_J._Betty'],
    ['Kotiaho_Janne', 'S.Kotiaho_Janne', 'kotiaho_janne'],
    ['Sica_V._Yanina', 'Sica_Yanina'],
    ['Cisneros-Montemayor_Andrés', 'Cisneros-Montemayor_AndrésM.'],
    ['Harmáčková_Zuzana', 'Harmáčková_ZuzanaV.', 'Harmáčková_V._Zuzana'],
    ['Chaplin-Kramer_Becky/Rebecca?', 'Chaplin-Kramer_Rebecca'],
    ['SimoSeixas_Cristiana', 'SimãoSeixas_Cristiana'],
    ['Stokland_B._HkonB.', 'Stokland_HåkonB.'],
    ['BoschPereira_Maureen_Laura', 'Pereira_Laura'],
]
# Same surname and given name, differing only by an initial or extra name.
SAME_PERSON_INITIALS = [
    ['Harrison_A._Paula', 'Harrison_Paula', 'harrison_paula'],
    ['Faith_P._Daniel', 'Faith_DanielP.', 'faith_daniel'],
    ['Holman_Ian', 'Holman_P._Ian'],
    ['Roy_Helen', 'Roy_E._Helen'],
    ['Rounsevell_Mark', 'Rounsevell_DanielAnthony_Mark'],
    ['Galil_Bella', 'Galil_S._Bella'],
    ['halmy_marwa', 'Halmy_MarwaA.'],
    ['barton_david', 'Barton_DavidN.'],
    ['eyster_harold', 'Eyster_N._Harold'],
    ['hiremath_ankila', 'Hiremath_J._Ankila'],
    ['tittensor_derek', 'Tittensor_DerekP.'],
    ['Alpizar_Francisco', 'Alpizar_Rodriguez_Francisco'],
    ['Mannetti_Lelani', 'Mannetti_M._Lelani'],
    ['Roskruge_Nicholas', 'Roskruge_RahiriTeAwherata_Nicholas'],
]
# Which URI survives a merge, where the automatic choice (see uri_quality) is wrong.
PREFERRED_URI = {'Stokland_HåkonB.', 'SimãoSeixas_Cristiana', 'Metzger_JeanPaul', 'Merçon_Juliana', 'Kotiaho_Janne'}
# Label for a merged person, where no record in the group has a good one.
LABEL_OVERRIDE = {'Stokland_HåkonB.': 'Håkon B. Stokland', 'SimãoSeixas_Cristiana': 'Cristiana Simão Seixas'}

# ---------------------------------------------------------------------------------

BLOCK = re.compile(r'^<([^>]+)>(.*?)(?=\n\n|\Z)', re.M | re.S)
PERSON_LIST = re.compile(r'foaf:Person ((?:<[^>]+>,\s+)*<[^>]+>)')
CHAPTER_OBJECTS = re.compile(r'ipbes:Chapter ((?:<[^>]+>,?\s*)+)')
CHAPTER_LIST = re.compile(r'\n    ipbes:Chapter ((?:<[^>]+>,\s+)*<[^>]+>) ;')


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))


def letters(s):
    return re.sub(r'[^a-z]', '', strip_accents(s).lower())


def words(s):
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', strip_accents(s))
    return frozenset(w for w in re.split(r'[^A-Za-z]+', s.lower()) if w)


def names_match(broken, candidate):
    """True if a broken URI's name plausibly is this person's name."""
    if letters(broken) == letters(candidate) or words(broken) == words(candidate):
        return True
    if words(broken) <= words(candidate) or words(candidate) <= words(broken):
        return True
    # Damaged names lose their non-ASCII letters: Babai_Dniel for Babai_Dániel
    if re.sub(r'[^a-z]', '', broken.lower()) == re.sub(r'[^a-z]', '', candidate.lower()):
        return True
    return difflib.SequenceMatcher(None, letters(broken), letters(candidate)).ratio() >= 0.85


def chapter_key(iri):
    """2-1, 2.1 and Ch2.1 compare equal, as in ReportController::fetchChapterContributors."""
    tail = re.sub(r'^.*ch', '', iri.rsplit('/', 1)[1])
    return re.sub(r'[^0-9]+', '.', tail).strip('.')


def is_person(subject, body):
    return subject.startswith(NS) and re.match(r'\s+a foaf:Person', body) is not None


def person_name(iri):
    return iri[len(NS):] if iri.startswith(NS) else None


def write_person_list(items):
    return 'foaf:Person ' + ',\n        '.join(f'<{i}>' for i in items)


def rewrite_person_lists(text, mapping, stats):
    """Apply {old name: new name, or None to remove} to every foaf:Person list."""
    def rewrite(match):
        items = re.findall(r'<([^>]+)>', match.group(1))
        if not any(person_name(i) in mapping for i in items):
            return match.group(0)
        out = []
        for item in items:
            name = person_name(item)
            if name in mapping:
                if mapping[name] is None:
                    stats['placeholder reference removed'] += 1
                    continue
                item = NS + mapping[name]
                stats['person reference repointed'] += 1
            if item not in out:
                out.append(item)
        if not out:
            sys.exit(f'A foaf:Person list would become empty: {match.group(0)[:200]}')
        return write_person_list(out)
    return PERSON_LIST.sub(rewrite, text)


def fix_broken_uris(texts, stats, notes):
    """Step 1: repoint foaf:Person references that have no person record."""
    typed_anywhere, retired = set(), {}
    for text in texts.values():
        for subject, body in BLOCK.findall(text):
            if is_person(subject, body):
                typed_anywhere.add(person_name(subject))
            replaced = re.match(r'\s+dcterms:isReplacedBy <([^>]+)>', body)
            if replaced and person_name(subject):
                retired[person_name(subject)] = person_name(replaced.group(1))

    for filename, text in texts.items():
        chapters_of, used_in = {}, collections.defaultdict(set)
        for subject, body in BLOCK.findall(text):
            if is_person(subject, body):
                found = CHAPTER_OBJECTS.search(body)
                chapters_of[person_name(subject)] = (
                    {chapter_key(i) for i in re.findall(r'<([^>]+)>', found.group(1))} if found else set())
            for found in PERSON_LIST.finditer(body):
                for item in re.findall(r'<([^>]+)>', found.group(1)):
                    if item.startswith(NS):
                        used_in[person_name(item)].add(chapter_key(subject))

        mapping = {}
        for name, keys in used_in.items():
            if name in typed_anywhere:
                continue
            if name in retired:
                mapping[name] = retired[name]
            elif name in PLACEHOLDER_REFERENCES:
                mapping[name] = None
            else:
                candidates = ([BROKEN_URI_TARGETS[name]] if name in BROKEN_URI_TARGETS
                              else [c for c in chapters_of if names_match(name, c)])
                confirmed = [c for c in candidates if c in chapters_of and chapters_of[c] & keys]
                if len(confirmed) == 1:
                    mapping[name] = confirmed[0]
                else:
                    notes['broken URIs left alone'].append(f'{filename}: {name} (candidates: {candidates})')
        texts[filename] = rewrite_person_lists(text, mapping, stats)


def read_people(texts):
    people = collections.defaultdict(lambda: {'orcid': set(), 'label': [], 'records': 0})
    for text in texts.values():
        for subject, body in BLOCK.findall(text):
            if is_person(subject, body):
                person = people[person_name(subject)]
                person['records'] += 1
                person['orcid'] |= {o for o in re.findall(r'https://orcid\.org/([^>]+)>', body)
                                    if not re.fullmatch(r'(?i)na|null', o)}
                person['label'] += re.findall(r'skos:prefLabel "([^"]*)"', body)
    return people


def uri_quality(name, people):
    """Sort key: the best URI in a merged group has the highest value."""
    person = people[name]
    label = (person['label'] or [''])[0]
    surname_first = bool(label) and letters(label.split()[-1]) in letters(name.split('_')[0])
    damaged = bool(re.search(r'[\xad\xa0?/()]|Ã|Å', name))
    given = name.split('_', 1)[1] if '_' in name else ''
    run_together = bool(re.search(r'[a-zà-ÿ][A-Z]', given))
    return (not damaged, name[0].isupper(), surname_first, not run_together,
            sum(ord(c) > 127 for c in name), len(person['orcid']), person['records'], len(name), name)


def plan_merges(people, notes):
    """Returns {name: surviving name} for every person in a merged group."""
    parent = {name: name for name in people}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def join(names):
        for name in names[1:]:
            parent[find(name)] = find(names[0])

    for key in (letters, words):
        grouped = collections.defaultdict(list)
        for name in people:
            if name not in PLACEHOLDER_RECORDS:
                grouped[key(name)].append(name)
        for names in grouped.values():
            join(names)
    for names in SAME_PERSON + SAME_PERSON_INITIALS:
        present = [n for n in names if n in people]  # merged names are gone after the first run
        if present:
            join(present)

    groups = collections.defaultdict(list)
    for name in people:
        groups[find(name)].append(name)

    survivor_of = {}
    for names in groups.values():
        if len(names) < 2:
            continue
        names = sorted(names)
        orcids = set().union(*[people[n]['orcid'] for n in names])
        if len(orcids) > 1:
            notes['not merged: same name, different ORCIDs'].append(' | '.join(names))
            continue
        preferred = [n for n in names if n in PREFERRED_URI]
        survivor = preferred[0] if preferred else max(names, key=lambda n: uri_quality(n, people))
        for name in names:
            survivor_of[name] = survivor
        others = ' | '.join(n for n in names if n != survivor)
        notes['merged, no ORCID to confirm' if not orcids else 'merged'].append(f'{survivor}  <=  {others}')
    return survivor_of


def surviving_values(survivor_of, people, texts):
    """The label and names every record of a merged person should carry."""
    values = {}
    for survivor in set(survivor_of.values()):
        values[survivor] = {}
        for text in texts.values():
            own = re.search(r'^<' + re.escape(NS + survivor) + r'> a foaf:Person(.*?)(?=\n\n|\Z)', text, re.M | re.S)
            if own:
                for predicate in ('skos:prefLabel', 'foaf:firstName', 'foaf:lastName'):
                    found = re.search(predicate + r' "([^"]*)"', own.group(1))
                    values[survivor][predicate] = found.group(1) if found else None
                break
        labels = [l for n, s in survivor_of.items() if s == survivor for l in people[n]['label']]
        if survivor in LABEL_OVERRIDE:
            values[survivor]['skos:prefLabel'] = LABEL_OVERRIDE[survivor]
        elif not values[survivor].get('skos:prefLabel') and labels:
            # The surviving record has no label: take the most complete one in the group
            values[survivor]['skos:prefLabel'] = max(labels, key=lambda l: (sum(ord(c) > 127 for c in l), len(l)))
    return values


def fix_records(text, survivor_of, values, group_has_orcid, stats):
    """Steps 2 and 3 on one file: chapter links, renames, retired-URI statements."""
    def fix_chapter_list(match):
        out = []
        for item in re.findall(r'<([^>]+)>', match.group(1)):
            tail = item[len(CH):] if item.startswith(CH) else None
            if tail in CHAPTER_DROP:
                stats['chapter link removed'] += 1
                continue
            if tail in CHAPTER_REPOINT:
                item = CH + CHAPTER_REPOINT[tail]
                stats['chapter link repointed'] += 1
            if item not in out:
                out.append(item)
        if not out:
            return ''
        return '\n    ipbes:Chapter ' + ',\n        '.join(f'<{i}>' for i in out) + ' ;'

    def fix_block(match):
        subject, body = match.group(1), match.group(2)
        if not is_person(subject, body):
            return match.group(0)
        name = person_name(subject)
        if name in PLACEHOLDER_RECORDS:
            stats['placeholder record removed'] += 1
            return '@@REMOVED@@'
        if BOM in body:
            body = body.replace(BOM, '')
            stats['stray BOM removed'] += 1
        body = CHAPTER_LIST.sub(fix_chapter_list, body)
        survivor = survivor_of.get(name)
        if survivor:
            for predicate, value in values[survivor].items():
                if value:
                    body = re.sub(predicate + r' "[^"]*"', lambda _, v=value: f'{predicate} "{v}"', body)
            if survivor in group_has_orcid:
                body, removed = re.subn(r'\n    owl:sameAs <https://orcid\.org/(?i:na|null)> ;', '', body)
                stats['placeholder ORCID removed'] += removed
        if survivor and survivor != name:
            stats['duplicate record merged'] += 1
            return f'<{NS}{survivor}>{body}\n\n<{NS}{name}> dcterms:isReplacedBy <{NS}{survivor}> .'
        return f'<{subject}>{body}'

    text = BLOCK.sub(fix_block, text)
    return text.replace('@@REMOVED@@\n\n', '').replace('\n\n@@REMOVED@@', '')


def report_remaining(texts, notes, survivor_of, people):
    orcid_owners = collections.defaultdict(set)
    for name, person in people.items():
        for orcid in person['orcid']:
            orcid_owners[orcid].add(survivor_of.get(name, name))
    for orcid, names in sorted(orcid_owners.items()):
        if len(names) > 1:
            notes['ORCID on more than one person (fix in the source)'].append(f'{orcid}: {" | ".join(sorted(names))}')

    dangling = collections.Counter()
    for text in texts.values():
        real = set(re.findall(r'^<(' + re.escape(CH) + r'[^>]+)> a ipbes:Chapter', text, re.M))
        for subject, body in BLOCK.findall(text):
            if is_person(subject, body):
                found = CHAPTER_OBJECTS.search(body)
                if found:
                    dangling.update(i[len(CH):] for i in re.findall(r'<([^>]+)>', found.group(1)) if i not in real)
    notes['chapter links with no chapter record'] += [f'{tail}: {n} people' for tail, n in sorted(dangling.items())]


def main():
    parser = argparse.ArgumentParser(description='Clean person and chapter links in the .ttl files.')
    parser.add_argument('--apply', action='store_true', help='write the changes (default: dry run)')
    parser.add_argument('--verbose', action='store_true', help='list every merge')
    parser.add_argument('--dir', type=pathlib.Path, default=DEFAULT_DIR, help='folder holding the .ttl files')
    args = parser.parse_args()

    paths = sorted(args.dir.glob('*.ttl'))
    if not paths:
        sys.exit(f'No .ttl files in {args.dir}')
    originals = {p.name: p.read_text(encoding='utf-8') for p in paths}
    texts = dict(originals)
    crlf = [name for name, text in texts.items() if '\r' in text]
    if crlf:
        sys.exit(f'Convert these to LF line endings first: {crlf}')

    stats = collections.Counter()
    notes = collections.defaultdict(list)

    fix_broken_uris(texts, stats, notes)
    people = read_people(texts)
    survivor_of = plan_merges(people, notes)
    values = surviving_values(survivor_of, people, texts)
    group_has_orcid = {s for n, s in survivor_of.items() if people[n]['orcid']}
    rename = {n: s for n, s in survivor_of.items() if n != s}
    for name in texts:
        text = fix_records(texts[name], survivor_of, values, group_has_orcid, stats)
        texts[name] = rewrite_person_lists(text, rename, stats)
    report_remaining(texts, notes, survivor_of, people)

    changed = [name for name in texts if texts[name] != originals[name]]
    print(f'Files that {"changed" if args.apply else "would change"}: {len(changed)} of {len(texts)}'
          + (f' ({", ".join(changed)})' if changed else ''))
    for key, count in sorted(stats.items()):
        print(f'  {key}: {count}')

    summary_only = {'merged', 'merged, no ORCID to confirm'}
    for title, lines in sorted(notes.items()):
        print(f'\n{title}: {len(lines)}')
        if args.verbose or title not in summary_only:
            for line in lines:
                print(f'  {line}')

    if args.apply:
        for path in paths:
            if path.name in changed:
                path.write_text(texts[path.name], encoding='utf-8', newline='')
        print('\nWritten. Next: review `git diff`, then `make load-data`.')
    else:
        print('\nDry run: nothing written. Re-run with --apply to write.')


if __name__ == '__main__':
    main()
