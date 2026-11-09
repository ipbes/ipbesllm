# IPBES report ontology: recommended changes from v06 to v07

**File:** `docs/ontology/IPBES_Ontology_v07.ttl`. These changes are proposed to the TSU on data for agreement. It recommends changes to v06.

**Basis:** 14 assessment reports (AFA, AMA, APA, BBA, ECA, GA1, IAS, LDR, NXS, PPA, SCM, SUA, TCA, VA) were converted automatically from their published PDFs to TTL and checked against v06. The files use ten properties v06 does not define: three on knowledge gaps that came from the reports, two links derived during conversion (changes 8 and 9), a single property for levels of confidence (change 10) and two roles, for the technical support unit and the editors (changes 11 and 12), a description on chapters (change 13) and a link from a merged person's old URI to the current one (change 14). Classes are added for guidance points (change 16), figures and tables (change 17) and footnotes (change 18); none is removed. Key and background messages record the section of the summary for policymakers they are printed in, and each section is a key message linked to them (change 19). 

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
| 12 | Add the role `ipbes:ed` to Person, for the editors on the title page | Yes | None: addition |
| 13 | Add `ipbes:hasDescription` to Chapter, for its executive summary | Yes | None: addition |
| 14 | Add `dcterms:isReplacedBy` to Person, for merged URIs | Yes | None: addition; published URIs keep working |
| 15 | Add `skos:note` to Person, for corrected ORCID iDs | Yes | None: addition |
| 16 | Add the class `ipbes:Guidance`, for guidance sections in a summary for policymakers | Yes | None: addition |
| 17 | Add `ipbes:Figure` and `ipbes:Table` as subclasses of `ipbes:Illustration`, with a link to the sub-chapter and the Zenodo DOI | Yes | None: illustrations keep their URIs and their type |
| 18 | Add the class `ipbes:Footnote` | Yes | None: addition; footnotes leave the sub-chapter text once converted |
| 19 | Add `ipbes:hasCategory` to KeyMessage and BackgroundMessage, and `ipbes:KeyMessage` to KeyMessage, for the sections of the summary for policymakers | Yes | None: additions; the section key messages of the manual files (NXS24-KM-A) gain links |
| 20–21 | Questions for discussion | No | See below |

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

**Other editors:** SCM's title page also lists other editors, by initials and with no role. They are recorded as editors (change 12).

### 12. Person: the editors

`ipbes:ed` records everyone listed under "Edited by:" on a report's title page (page 3), with values such as "Editor in NXS24". The editors are the co-chairs and the technical support unit, and in some reports also members of the Multidisciplinary Expert Panel and other experts: SCM lists 19, among them an expert panel member and 15 experts given by initials, who are matched to the chapter authors. Co-chairs and the technical support unit keep their own role (`cs`, `ts`) as well. Editors are also linked from the Report with `foaf:Person`.

The converted files have 71 of these values. The manual files do not record the editors.

**The management committee is not recorded.** An earlier draft of v07 had a role `ipbes:mc` for the management committee listed on page 2 (Bureau and Multidisciplinary Expert Panel). It is replaced by `ipbes:ed`: the committee guided the report but did not write it, and those of its members who edited the report are on page 3.

**Point to note:** the same person can be written differently across reports ("Robert Watson", "Robert T. Watson"), and so have different URIs.

### 13. Chapters: the executive summary

`ipbes:hasDescription` on a Chapter holds its executive summary, finding by finding, with its traceability marks and footnotes as printed. v06 has no place for it, so it was left out of the files, which lost text the reports give as each chapter's main findings.

### 14. Person: URIs that are merged

When records turn out to be the same person, they are merged under one URI. The old URI stays, with only `dcterms:isReplacedBy` pointing to the URI that survives, so anything that already cites the old URI can still be resolved. A retired URI is no longer typed as a Person.

Merging follows reviewed rules: records with different ORCID iDs are never merged, and a shared ORCID iD alone is never a reason to merge.

### 15. Person: a note on a corrected ORCID iD

Some assessments entered one ORCID iD for several people. The IPBES experts list is the verified source. `skos:note` on a Person says what the assessment's records gave, when the iD now differs from it or was removed. It is used on 7 person records (GA1, NXS, SUA, TCA, VA). A person who rightly holds the iD gets no note.

### 16. A class for guidance points

The summary for policymakers of the scenarios and models assessment (SCM) ends with two sections: "Guidance for science and policy" (guidance points 1 to 6) and "Guidance for the Platform and its task forces and expert groups" (Platform guidance points 1 to 6). They are neither key messages nor background messages, and v06 has no place for them, so they were left out of the manual file and their text would otherwise be lost, although the key findings point to them ("See guidance point 3").

`ipbes:Guidance` follows `ipbes:KnowledgeGap`: `hasCategory` holds the section, `hasDescription` the text, `traceability` and `SubChapter` what the report cites. `skos:prefLabel` holds the printed label and `dcterms:identifier` the report code with it (`SCM16-GP1`, `SCM16-PGP1`). `skos:note` holds the sentence printed above the first section's points. SCM is the only one of the 14 reports with such points, but the class is not specific to it.

### 17. Figures and tables as classes of their own

`ipbes:Figure` and `ipbes:Table` are subclasses of `ipbes:Illustration`, so figures, tables and boxes can be listed and displayed separately. Boxes stay `ipbes:Illustration` only.

**URIs do not change.** Figures and tables keep their `il/` URIs (`il/IAS23-Figure3.9`), so the links from messages and sub-messages, and anything that already cites them, keep working. Each is typed with both classes, `a ipbes:Figure, ipbes:Illustration`, because IBOK and other queries that use no reasoner would otherwise stop finding them as illustrations.

Three properties of `ipbes:Illustration` are added or defined for this:

| Property | Holds | Source |
|---|---|---|
| `ipbes:SubChapter` | The sub-chapter the item is printed in. None in a summary for policymakers | The conversion knows it but the files do not record it yet |
| `ipbes:hasDoi` | The concept DOI of the Zenodo record holding the item's files, which resolves to the latest version (`10.5281/zenodo.8041593`). It identifies the chapter's record, not the single item | The records listed in `data/csv/figures_tables.csv`, and the sibling records of the other chapters |
| `ipbes:hasDescription` | The caption's text after its title, from the caption file in the Zenodo package where there is one | Caption files exist for IAS and TCA; the PDF legend otherwise |

`skos:prefLabel` keeps the title and `dcterms:identifier` the kind and number (`Figure3.9`, `TableSPM.A1`).

**Images are not part of the ontology.** IBOK keeps the image files as a resource in GitHub and finds each one from the item's identifier, by the naming rule under Practices to confirm. No property points to a file.

**Coverage:** of 2,389 figures and tables in the 14 reports, Zenodo has an SVG, JPEG or PNG for 320 (IAS, TCA, VA, SUA, BBA) and a PDF only for about 178. The others are only in the report PDFs.

### 18. A class for footnotes

`ipbes:Footnote` holds the footnotes, which the files now keep at the end of their sub-chapter's text under "Notes".

| Property | Holds |
|---|---|
| `dcterms:identifier` | Footnote, its chapter and its number as printed: `Footnote1.12` is note 12 of chapter 1, `FootnoteSPM.3` note 3 of the summary for policymakers. Numbering restarts in each chapter |
| `ipbes:hasDescription` | The text of the footnote |
| `ipbes:SubChapter` | The sub-chapter it belongs to |
| `ipbes:Report` | The report |

The URI follows the illustrations: `fn/IAS23-Footnote1.12`. The 14 reports have about 570 numbered footnotes in their sub-chapters' "Notes", most in TCA (115), VA (88) and ECA (78).

### 19. Sections of the summary for policymakers

The key and background messages of most summaries for policymakers are printed under lettered sections, such as "A. Past and current nexus interactions" and "B. Future nexus interactions" (NXS). The files did not record them, so a key message could not be found from its section.

| Where | Property | Holds |
|---|---|---|
| KeyMessage, BackgroundMessage | `ipbes:hasCategory` | The section it is printed in, with its letter: "A. Past and current nexus interactions" |
| KeyMessage (a section) | `ipbes:KeyMessage` | The key messages printed in the section |
| KeyMessage (a section) | `ipbes:BackgroundMessage` | The background messages printed in the section |

Each section is a key message, as in the manual files of IAS, NXS and TCA: `key/NXS24-KM-A`, with `dcterms:identifier` "A" and the heading as `skos:prefLabel`. Where the key messages are numbered KM1, KM-A1 …, the section's URI is `KM-` and its letter; otherwise the letter alone (`key/AFA18-A`). The section links down to its messages, as the Report and Chapter link to their people, so opening it lists them. The manual files have the sections but no such links.

The converted files have 42 sections in 12 reports. ECA and SUA need none: their key messages A, B … are the sections. SCM has none. BBA and VA print their key messages without sections (each cites background messages from several), so their sections link to background messages only.

**hasCategory** is already used on KnowledgeGap and Guidance for the heading an item is printed under, so no new property is needed.

## Questions for discussion (not in v07)

**20. Chapters printed in parts.** GA1's chapter 2 is printed in parts 2.1, 2.2 and 2.3, each with its own authors and reference list. The automatic file has one chapter with sub-chapters 2.1.1 ...; the manual file has three chapters. Which should be the model?

**21. Properties defined but unused in the converted files.**

| Property | Note |
|---|---|
| `owl:sameAs` on Illustration | Figures, tables and boxes have no other resource to link to. (`hasDoi` is now used for the Zenodo records, change 17) |
| `hasConfidenceLevel` on KeyMessage | No key message headline states one in the form "(well established)" |

Should these stay as optional?

## Practices to confirm

These need no change to the ontology, but other producers of TTL files should follow them:

- **Sub-messages** are made of full sentences. A new one starts at each sentence stating a level of confidence; a sentence without a level joins its neighbour. A level is the minimum, and an identifier is added where the report gives a traceability mark.
- **Knowledge gaps** are numbered from 1, in the order printed.
- **Image files** of figures and tables are named after the local name of the item's URI, so the image of `il/IAS23-Figure3.9` is `IAS23-Figure3.9.svg` and of `il/TCA24-FigureSPM.5` is `TCA24-FigureSPM.5.svg`: the URI alone finds the file. An item with several images adds `_1`, `_2` …, and a part printed in the source file's name adds its letter (`VA22-Table2.6_A.jpg`).
- **DOIs** are the ones that resolve to the latest version, not those of a specific version. The DOIs of the 2016 to 2019 reports were created after publication and are not printed in them. The reviewed list is kept in the conversion's configuration.
- **Key messages and background messages:** where a report prints which background messages support a key message, or numbers them under it, those are linked. AFA, AMA, APA, GA1, LDR and PPA print no such link, so a key message is linked to all the background messages of its section (A, B ...). In those reports the link means "same section", and through it a sub-chapter links to all of that section's key messages.
