# IPBES report ontology: recommended changes from v06 to v07

**File:** `docs/ontology/IPBES_Ontology_v07.ttl`. These changes are proposed to the TSU on data for agreement. It recommends changes to v06.

**Basis:** 14 assessment reports (AFA, AMA, APA, BBA, ECA, GA1, IAS, LDR, NXS, PPA, SCM, SUA, TCA, VA) were converted automatically from their published PDFs to TTL and checked against v06. The files use nine properties v06 does not define: three on knowledge gaps that came from the reports, two links derived during conversion (changes 8 and 9), a single property for levels of confidence (change 10) and two roles, for the technical support unit and the management committee (changes 11 and 12), and a description on chapters (change 13). No class is added or removed. 

## Summary of decisions requested

| # | Change | In v07 | Effect on existing data |
|---|---|---|---|
| 1 | Namespace with a trailing slash | Yes | All URIs of classes and properties; manual files and IBOK queries |
| 2 | Add `ipbes:hasCategory` and `ipbes:traceability` to KnowledgeGap | Yes | None: additions |
| 3 | Add `skos:note` to KnowledgeGap | Yes | None: addition |
| 4 | Redefine `dcterms:identifier` on KnowledgeGap | Yes | Older manual files hold sections there |
| 5 | Reword `dcterms:identifier` on SubMessage | Yes | None: wording matches practice |
| 6 | Reword `ipbes:country` on Person | Yes | None: wording matches the reports |
| 7 | Correct typing errors in descriptions | Yes | None |
| 8 | Add `ipbes:SubChapter` to SubChapter, as a link to its parent | Yes | None: addition |
| 9 | Add `ipbes:BackgroundMessage` to KnowledgeGap | Yes | None: addition; the manual GA1 file already has it |
| 10 | Replace the four level-of-confidence properties with `ipbes:hasConfidenceLevel` | Yes | Manual files and IBOK queries use the four old properties |
| 11 | Add the role `ipbes:ts` to Person, for the technical support unit | Yes | None: addition |
| 12 | Add the role `ipbes:mc` to Person, for the management committee | Yes | None: addition |
| 13 | Add `ipbes:hasDescription` to Chapter, for its executive summary | Yes | None: addition |
| 14–15 | Questions for discussion | No | See below |

## Changes in v07

### 1. Namespace with a trailing slash

| | Namespace | Example class |
|---|---|---|
| v06, manual conversions | `http://ontology.ipbes.net/report` | `http://ontology.ipbes.net/reportKeyMessage` |
| v07, automatic conversions | `http://ontology.ipbes.net/report/` | `http://ontology.ipbes.net/report/KeyMessage` |

**Why:** without the slash, the prefix runs into the term, giving `reportKeyMessage`. IBOK has had to be adapted to handle the manual files, which lack the slash. With the slash in the ontology and the files, that adaptation is no longer needed.

**Action needed:** this has to change together in the ontology, the files and IBOK. SPARQL queries declaring `PREFIX ipbes: <http://ontology.ipbes.net/report>` will not match the new files until the prefix is updated.

### 2. Knowledge gaps: category and traceability

| Property | Meaning | Use in the converted files |
|---|---|---|
| `ipbes:hasCategory` | The gap's category, as printed in the report | 310 of 310 gaps |
| `ipbes:traceability` | The sections, figures, tables or boxes given for the gap | 226 of 310 gaps |

The other gaps have no traceability because their reports print none for them (for example GA1's Appendix 4 and LDR's Table SPM 3). The latest manual conversions (BBA, GA1, NXS) already use both properties.

### 3. Knowledge gaps: a note on how a gap was obtained

`skos:note` records how a gap was obtained when the report does not list it as one. It is used on 32 gaps, all in VA.

VA lists 8 gaps in Table SPM 5 and discusses others in running text in section 6.4.4. Those 32 were extracted systematically, one per paragraph of 6.4.4.1 to 6.4.4.9, in the report's own words. The note lets users tell them apart from listed gaps. 

### 4. Knowledge gaps: what the identifier holds

| | `dcterms:identifier` on KnowledgeGap |
|---|---|
| v06 description | The sub-chapter or illustration mentioned in the description |
| v07 description | The gap's own identifier: the report code and its number from 1, as in its URI (`BBA26-1`) |

**Why:** this matches the latest manual conversions (BBA, GA1, NXS), and every gap then has an identifier, including those with no sections. The sections move to `ipbes:traceability`. The older manual files (IAS, SUA, TCA, VA) hold sections in this property and number gaps from 0.

### 5. Sub-messages: what the identifier holds

The description now says what the property holds in practice: the sections, figures, tables or boxes the sub-message traces to, not "the identifier of the chapter or sub-chapter or key message".

### 6. Person: country

The description changes from "Country of residence of the person" to the country as printed in the report (citizenship and affiliation), as an ISO 3166 alpha-3 code. The reports do not state residence separately.

### 7. Typing errors

"knowlege" and "bakcgound" are corrected, and "in the stated the chapter and report" becomes "in the stated chapter and report".

### 8. Sub-chapters: a link to the parent

`ipbes:SubChapter` on a SubChapter points to its parent: the nearest numbered section above it that exists (3.1.1 to 3.1). Until now the hierarchy could be read only from the identifier. A top section (3.1) has no parent sub-chapter, only its `ipbes:Chapter`.

The converted files have 3,963 of these links. Each was checked against the identifiers, and none is wrong.

**Point to note:** the property has the same name as the links from sub-messages, knowledge gaps and references to their sub-chapters, following the ontology's habit of naming a link after its target class. Queries on `ipbes:SubChapter` that do not filter by the subject's class will now also return sub-chapters.

### 9. Knowledge gaps: a link to background messages

`ipbes:BackgroundMessage` on a KnowledgeGap points to each background message with a sub-message tracing to one of the gap's own sections. The reports do not print this relation, so the link means "shares a section", not "the report says this message concerns this gap".

| | Gaps | Linked | Links per linked gap (median, maximum) |
|---|---|---|---|
| BBA | 37 | 36 | 6, 20 |
| ECA | 7 | 5 | 2, 3 |
| IAS | 40 | 32 | 2, 9 |
| NXS | 30 | 12 | 1, 4 |
| SUA | 61 | 44 | 3.5, 9 |
| TCA | 20 | 19 | 9, 18 |
| VA | 40 | 6 | 1, 6 |
| GA1 | 58 | 0 | No sections printed for its gaps |
| LDR | 17 | 0 | No sections printed for its gaps |

In all, 154 of the 310 gaps are linked. A gap that cites many sections can link to many messages, so a high count shows breadth rather than a close relation.

**Differs from the manual GA1 file:** it has this link on 15 gaps, each pointing to A4, which the automatic file does not reproduce.

### 10. One property for the level of confidence

`hasWellestablished`, `hasEstablishedIncomplete`, `hasUnresolved` and `hasInconclusive`, each on KeyMessage, BackgroundMessage and SubMessage, are replaced by one property:

| Property | Values |
|---|---|
| `ipbes:hasConfidenceLevel` | `well established`, `established but incomplete`, `unresolved`, `inconclusive` |

**Why:** the old properties repeated their own names as values: every `hasWellestablished` held "well established". One property takes 12 lines out of the ontology and shortens queries, and a new level would need only a new value.

**Several levels:** the property is repeated when an item states more than one. In the converted files, 2,215 items have 2,285 values, and 69 items state more than one.

**Action needed:** like the namespace, this has to change together in the ontology, the files and IBOK. The manual files keep the old properties, and queries on them will not match the new files.

**Values:** they are plain text for now. Making them concepts with URIs and definitions is a possible later step.

### 11. Person: the technical support unit

`ipbes:ts` records the technical support unit members listed as editors on a report's title page, with values such as "Technical Support Unit in PPA16". The name follows the two-letter roles (`cs`, `cl`, `la`, `re`, `fl`, `ca`) and the term the reports use. Like co-chairs, these people are also linked from the Report with `foaf:Person`.

The converted files have 18 of these values. BBA and TCA list none on the title page. The manual files do not record the technical support unit.

**Other editors:** SCM's title page also lists other editors by initials, with no role. No role is recorded from that page for them.

### 12. Person: the management committee

`ipbes:mc` records the members of the management committee listed on page 2 of each report, with values such as "Management Committee in PPA16". The earlier reports divide them into Bureau and Multidisciplinary Expert Panel, and the later ones list them together, so the one role covers both. They are also linked from the Report with `foaf:Person`.

The members come from page 2 of each report and from the experts list, which gives their full names, countries and ORCID iDs. Anyone found in only one of the two is kept and noted in the conversion log.

**Point to note:** the same person can be written differently across reports ("Robert Watson", "Robert T. Watson"), and so have different URIs.

### 13. Chapters: the executive summary

`ipbes:hasDescription` on a Chapter holds its executive summary, finding by finding, with its traceability marks and footnotes as printed. v06 has no place for it, so it was left out of the files, which lost text the reports give as each chapter's main findings.

## Questions for discussion (not in v07)

**14. Chapters printed in parts.** GA1's chapter 2 is printed in parts 2.1, 2.2 and 2.3, each with its own authors and reference list. The automatic file has one chapter with sub-chapters 2.1.1 ...; the manual file has three chapters. Which should be the model?

**15. Properties defined but unused in the converted files.**

| Property | Note |
|---|---|
| `hasDoi`, `owl:sameAs` on Illustration | Figures, tables and boxes have neither in the reports |
| `hasConfidenceLevel` on KeyMessage | No key message headline states one in the form "(well established)" |

Should these stay as optional?

## Practices to confirm

These need no change to the ontology, but other producers of TTL files should follow them:

- **Sub-messages** are made of full sentences. A new one starts at each sentence stating a level of confidence; a sentence without a level joins its neighbour. A level is the minimum, and an identifier is added where the report gives a traceability mark.
- **Knowledge gaps** are numbered from 1, in the order printed.
- **DOIs** are the ones that resolve to the latest version, not those of a specific version. The DOIs of the 2016 to 2019 reports were created after publication and are not printed in them. The reviewed list is kept in the conversion's configuration.
- **Key messages and background messages:** where a report prints which background messages support a key message, or numbers them under it, those are linked. AFA, AMA, APA, GA1, LDR and PPA print no such link, so a key message is linked to all the background messages of its section (A, B ...). In those reports the link means "same section", and through it a sub-chapter links to all of that section's key messages.
