# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for the `README.md` of driverisk and for this file. Section 3 gives the project
vocabulary. Each term in Section 3 has one meaning in all of the documentation.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the driverisk documentation. The code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **reading** | One row of the long telematics stream: device, time, variable, value | record, sample, message |
| **signal** | One measured quantity after the ingest, for example `speed_kmh` | variable (after the ingest), channel |
| **device** | One vehicle unit, the `deviceId`. The project uses one device for one driver | car, unit, user |
| **trip** | Readings of one device between an ignition start and an ignition stop or a long gap | journey, drive, session |
| **trip table** | The table with one row for each trip: features, exposure and target | dataset (alone), feature matrix |
| **exposure** | The trip distance in km | mileage, weight (in prose) |
| **harsh event** | One run of readings over a braking, acceleration or lateral limit | incident, alert, alarm |
| **target** | The count of harsh events in a trip | label, outcome |
| **rate** | Harsh events per km | frequency, intensity |
| **context** | Weather, road class and collision density at the place and hour of a reading | environment, enrichment |
| **cell** | A square of the map: 0.1° for weather, 0.02° for offline roads, 0.01° for collisions | tile, grid box |
| **provider** | A class that gives weather or road class: offline, CSV or API | source (for a class), backend |
| **collision prior** | Collisions for each cell from a collision table | accident map, hot spot |
| **fleet rate** | Total events divided by total km of the training trips | average rate, base rate |
| **held-out devices** | The 25 % of devices that `train` keeps out until the final score | test set, holdout drivers |
| **risk index** | 100 × the predicted rate of a driver divided by the fleet mean | score (alone), rating |
| **run folder** | The folder with `model.joblib`, `metrics.json` and `model_card.md` | output, artefact folder |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **ingest** | Change the long stream into one typed row for each device and second |
| **segment** | Divide the readings of a device into trips |
| **attach** | Add context columns to readings by cell and hour |
| **count** | Find the harsh events of a trip |
| **fit** | Learn model parameters from the trips of the training devices |
| **score** | Give a predicted rate and a risk index to each driver |
