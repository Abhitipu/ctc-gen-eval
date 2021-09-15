# CTC Score

## Installation
```
pip install -e .
```

## Example
```python
from ctc_score import StyleTransferScorer, SummarizationScorer, DialogScorer

# Topical-Chat
dialog_history = "so, i'm reading the latest film from studio ghibli is out the tale of princess kaguya. dunno if you're familiar with them, but studio ghibli has made a lot of great animated films, like spirited away, and princess mononoke \n i don't think i have heard of them. i have heard that one of the directors recently passed away, and his last film was nominated for an academy award \n yeah, sadly, disney ( which owns the american rights to the films ) doesn't tend to promote them very much. i think they're worried they 'll cut into their \" home grown \" market. anyway, dunno if you even like animated movies, but they're worth checking out. \n i don't watch them very often. apparently there was a showing of the recent film in a park in d.c. that's one u.s. city i haven't been to \n sadly, i haven't been to dc either, although i've always wanted to visit there. apparently there's a lot of interesting going down this summer. they're having a crab feast at the navy - marine corps stadium. they 'll have 100 gallons of crab soup! can you imagine that much soup? \n\n"
hypo = "i recently met a girl who lives in that area, and she said the nightlife is worth visiting for. it sounds like many of the events feature jazz music. do you listen to jazz very often?"
fact = "from left, emma baker, daniel saperstein and taylor mulitz of flasher will perform this summer's final fort reno concert. ( jared soares for the washington post ) monday, july 30 25th birthday celebration at national postal museum : celebrate 25 years of this institution devoted to the long history of the u.s. postal service with daytime festivities that include cupcakes, birthday postcards, a photo booth and a special scavenger hunt with prizes. 11 a.m. to 2 p.m. free. tuesday, july 31 \" the color purple \" at kennedy center : the tony award - winning musical revival, based on the pulitzer prize - winning alice walker novel of the same name, features jazz, ragtime, gospel and blues with a story about an african american woman named celie surviving poverty in the south during the 1930s. through aug. 26. $ 69-$149. ask a harry potter scholar at southeast neighborhood library : come to this talk from tolanda henderson, a librarian from george washington university, who has used the j.k. rowling book series as a text in academia. commune with other muggles who prove that it's not just kids and young adults who obsess about the boy who lived. 7 p.m. free. wednesday, aug. 1 rico nasty at the fillmore silver spring : two summers ago, rico nasty was a teenage loudmouth from the maryland suburbs, generating buzz on youtube for spitting surly, rainbow - tinted rhymes. now, after signing a deal with atlantic records, the 21-year - old singer is on her way to becoming one of the brightest voices in rap music.\n"

scorer = DialogScorer(align='D-topical_chat')

score = scorer.score(fact=fact, dialog_history=dialog_history, hypo=hypo, aspect='engagingness')
print(score)

# SummEval
doc = "(CNN)Donald Sterling's racist remarks cost him an NBA team last year. But now it's his former female companion who has lost big. A Los Angeles judge has ordered V. Stiviano to pay back more than $2.6 million in gifts after Sterling's wife sued her. In the lawsuit, Rochelle \"Shelly\" Sterling accused Stiviano of targeting extremely wealthy older men. She claimed Donald Sterling used the couple's money to buy Stiviano a Ferrari, two Bentleys and a Range Rover, and that he helped her get a $1.8 million duplex. Who is V. Stiviano? Stiviano countered that there was nothing wrong with Donald Sterling giving her gifts and that she never took advantage of the former Los Angeles Clippers owner, who made much of his fortune in real estate. Shelly Sterling was thrilled with the court decision Tuesday, her lawyer told CNN affiliate KABC. \"This is a victory for the Sterling family in recovering the $2,630,000 that Donald lavished on a conniving mistress,\" attorney Pierce O'Donnell said in a statement. \"It also sets a precedent that the injured spouse can recover damages from the recipient of these ill-begotten gifts.\" Stiviano's gifts from Donald Sterling didn't just include uber-expensive items like luxury cars. According to the Los Angeles Times, the list also includes a $391 Easter bunny costume, a $299 two-speed blender and a $12 lace thong. Donald Sterling's downfall came after an audio recording surfaced of the octogenarian arguing with Stiviano. In the tape, Sterling chastises Stiviano for posting pictures on social media of her posing with African-Americans, including basketball legend Magic Johnson. \"In your lousy f**ing Instagrams, you don't have to have yourself with -- walking with black people,\" Sterling said in the audio first posted by TMZ. He also tells Stiviano not to bring Johnson to Clippers games and not to post photos with the Hall of Famer so Sterling's friends can see. \"Admire him, bring him here, feed him, f**k him, but don't put (Magic) on an Instagram for the world to have to see so they have to call me,\" Sterling said. NBA Commissioner Adam Silver banned Sterling from the league, fined him $2.5 million and pushed through a charge to terminate all of his ownership rights in the franchise. Fact check: Donald Sterling's claims vs. reality CNN's Dottie Evans contributed to this report."
hypo = "-lrb- cnn -rrb- donald sterling 's racist remarks cost him an nba team last year . but now it 's his former female companion who has lost big . a los angeles judge has ordered v. stiviano to pay back more than $ 2.6 million in gifts after sterling 's wife sued her ."

scorer = SummarizationScorer(align='D-cnndm')

score = scorer.score(doc=doc, refs=[], hypo=hypo, aspect='consistency')
print(score)

# Yelp
input_sent = "i complained and received a complimentary room for one night ."
hypo = "i complained and received a great deal for date night ."
scorer = StyleTransferScorer(align='E-roberta')

score = scorer.score(input_sent=input_sent, hypo=hypo, aspect='preservation')
print(score)
```