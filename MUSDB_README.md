# MUSDB 2018 Dataset

The _musdb18_ consists of 150 songs of different styles along with the images of their constitutive objects.

_musdb18_ contains two folders, a folder with a training set: "train", composed of 100 songs, and a folder with a test set: "test", composed of 50 songs. Supervised approaches should be trained on the training set and tested on both sets.

All files from the _musdb18_ dataset are encoded in the [Native Instruments stems format](http://www.stems-music.com/) (.mp4). It is a multitrack format composed of 5 stereo streams, each one encoded in AAC @256kbps. These signals correspond to:

- `0` - The mixture,
- `1` - The drums,
- `2` - The bass,
- `3` - The rest of the accompaniment,
- `4` - The vocals.

For each file, the mixture correspond to the sum of all the signals. All signals are stereophonic and encoded at 44.1kHz.

The data from _musdb18_ is composed of several different sources:

* 100 tracks are taken from the [DSD100 dataset](http://sisec17.audiolabs-erlangen.de/#/dataset), which is itself derived from [The 'Mixing Secrets' Free Multitrack Download Library](www.cambridge-mt.com/ms-mtk.htm). Please refer to this original resource for any question regarding your rights on your use of the DSD100 data.
* 46 tracks are taken from [the MedleyDB](http://medleydb.weebly.com) licensed under Creative Commons (BY-NC-SA 4.0).
* 2 tracks were kindly provided by Native Instruments originally part of [their stems pack](https://www.native-instruments.com/en/specials/stems-for-all/free-stems-tracks/).
* 2 tracks a from from the Canadian rock band The Easton Ellises as part of the [heise stems remix competition](https://www.heise.de/ct/artikel/c-t-Remix-Wettbewerb-The-Easton-Ellises-2542427.html#englisch), licensed under Creative Commons (BY-NC-SA 3.0).

### List of tracks and license

- A Classic Education - NightOwl,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Actions - Devil's Words,Pop/Rock,DSD,Restricted
- Actions - One Minute Smile,Pop/Rock,DSD,Restricted
- Actions - South Of The Water,Pop/Rock,DSD,Restricted
- Aimee Norwich - Child,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Al James - Schoolboy Facination,Pop/Rock,DSD,Restricted
- Alexander Ross - Goodbye Bolero,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Alexander Ross - Velvet Curtain,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- AM Contra - Heart Peripheral,Pop/Rock,DSD,Restricted
- Angela Thomas Wade - Milk Cow Blues,Country,DSD,Restricted
- ANiMAL - Clinic A,Rap,DSD,Restricted
- ANiMAL - Easy Tiger,Rap,DSD,Restricted
- ANiMAL - Rockshow,Rap,DSD,Restricted
- Arise - Run Run Run,Reggae,DSD,Restricted
- Atlantis Bound - It Was My Fault For Waiting,Pop/Rock,DSD,Restricted
- Auctioneer - Our Future Faces,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- AvaLuna - Waterduct,Rock,MedleyDB,CC BY-NC-SA
- Ben Carrigan - We'll Talk About It All Tonight,Pop/Rock,DSD,Restricted
- BigTroubles - Phantom,Rock,MedleyDB,CC BY-NC-SA
- Bill Chudziak - Children Of No-one,Pop/Rock,DSD,Restricted
- BKS - Bulldozer,Pop/Rock,DSD,Restricted
- BKS - Too Much,Pop/Rock,DSD,Restricted
- Black Bloc - If You Want Success,Pop/Rock,DSD,Restricted
- Bobby Nobody - Stitch Up,Pop/Rock,DSD,Restricted
- Buitraker - Revo X,Pop/Rock,DSD,Restricted
- Carlos Gonzalez - A Place For Us,Pop/Rock,DSD,Restricted
- Celestial Shore - Die For Us,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Chris Durban - Celebrate,Electronic,DSD,Restricted
- Clara Berry And Wooldog - Air Traffic,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Clara Berry And Wooldog - Stella,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Clara Berry And Wooldog - Waltz For My Victims,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Cnoc An Tursa - Bannockburn,Heavy Metal,DSD,Restricted
- Creepoid - OldTree,Rock,MedleyDB,CC BY-NC-SA
- Cristina Vane - So Easy,Pop/Rock,DSD,Restricted
- Dark Ride - Burning Bridges,Heavy Metal,DSD,Restricted
- Detsky Sad - Walkie Talkie,Pop/Rock,DSD,Restricted
- Dreamers Of The Ghetto - Heavy Love,Pop,MedleyDB,CC BY-NC-SA
- Drumtracks - Ghost Bitch,Pop/Rock,DSD,Restricted
- Enda Reilly - Cur An Long Ag Seol,Pop/Rock,DSD,Restricted
- Faces On Film - Waiting For Ga,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Fergessen - Back From The Start,Pop/Rock,DSD,Restricted
- Fergessen - Nos Palpitants,Pop/Rock,DSD,Restricted
- Fergessen - The Wind,Pop/Rock,DSD,Restricted
- Flags - 54,Pop/Rock,DSD,Restricted
- Forkupines - Semantics,Pop/Rock,DSD,Restricted
- Georgia Wonder - Siren,Pop/Rock,DSD,Restricted
- Girls Under Glass - We Feel Alright,Electronic,DSD,Restricted
- Giselle - Moss,Electronic,DSD,Restricted
- Grants - PunchDrunk,Rap,MedleyDB,CC BY-NC-SA
- Helado Negro - Mitad Del Mundo,Pop,MedleyDB,CC BY-NC-SA
- Hezekiah Jones - Borrowed Heart,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Hollow Ground - Ill Fate,Heavy Metal,DSD,Restricted
- Hollow Ground - Left Blind,Heavy Metal,DSD,Restricted
- Hop Along - Sister Cities,Rock,MedleyDB,CC BY-NC-SA
- Invisible Familiars - Disturbing Wildlife,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- James Elder & Mark M Thompson - The English Actor,Heavy Metal,DSD,Restricted
- James May - All Souls Moon,Pop/Rock,DSD,Restricted
- James May - Dont Let Go,Pop/Rock,DSD,Restricted
- James May - If You Say,Pop/Rock,DSD,Restricted
- James May - On The Line,Pop/Rock,DSD,Restricted
- Jay Menon - Through My Eyes,Pop/Rock,DSD,Restricted
- Johnny Lokke - Promises & Lies,Pop/Rock,DSD,Restricted
- Johnny Lokke - Whisper To A Scream,Pop/Rock,DSD,Restricted
- Jokers Jacks & Kings - Sea Of Leaves,Pop/Rock,DSD,Restricted
- Juliet's Rescue - Heartbeats,Pop/Rock,DSD,Restricted
- Leaf - Come Around,Pop/Rock,DSD,Restricted
- Leaf - Summerghost,Pop/Rock,DSD,Restricted
- Leaf - Wicked,Pop/Rock,DSD,Restricted
- Little Chicago's Finest - My Own,Rap,DSD,Restricted
- Louis Cressy Band - Good Time,Pop/Rock,DSD,Restricted
- Lushlife - Toynbee Suite,Rap,MedleyDB,CC BY-NC-SA
- Lyndsey Ollard - Catching Up,Pop/Rock,DSD,Restricted
- M.E.R.C. Music - Knockout,Rap,DSD,Restricted
- Matthew Entwistle - Dont You Ever,Jazz,MedleyDB,CC BY-NC-SA
- Meaxic - Take A Step,Rock,MedleyDB,CC BY-NC-SA
- Meaxic - You Listen,Rock,MedleyDB,CC BY-NC-SA
- Moosmusic - Big Dummy Shake,Pop/Rock,DSD,Restricted
- Motor Tapes - Shore,Pop/Rock,DSD,Restricted
- Mu - Too Bright,Pop/Rock,DSD,Restricted
- Music Delta - 80s Rock,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Beatles,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- Music Delta - Britpop,Pop,MedleyDB,CC BY-NC-SA
- Music Delta - Country1,Country,MedleyDB,CC BY-NC-SA
- Music Delta - Country2,Country,MedleyDB,CC BY-NC-SA
- Music Delta - Disco,Pop,MedleyDB,CC BY-NC-SA
- Music Delta - Gospel,Pop,MedleyDB,CC BY-NC-SA
- Music Delta - Grunge,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Hendrix,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Punk,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Reggae,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Rock,Rock,MedleyDB,CC BY-NC-SA
- Music Delta - Rockabilly,Rock,MedleyDB,CC BY-NC-SA
- Nerve 9 - Pray For The Rain,Pop/Rock,DSD,Restricted
- Night Panther - Fire,Pop,MedleyDB,CC BY-NC-SA
- North To Alaska - All The Same,Pop/Rock,DSD,Restricted
- Patrick Talbot - A Reason To Leave,Jazz,DSD,Restricted
- Patrick Talbot - Set Free Me,Jazz,DSD,Restricted
- Phre The Eon - Everybody's Falling Apart,Pop/Rock,DSD,Restricted
- Port St Willow - Stay Even,Singer/Songwriter,MedleyDB,CC BY-NC-SA
- PR - Happy Daze,Electronic,Native Instruments,Restricted
- PR - Oh No,Electronic,Native Instruments,Restricted
- Punkdisco - Oral Hygiene,Pop/Rock,DSD,Restricted
- Raft Monk - Tiring,Pop/Rock,DSD,Restricted
- Remember December - C U Next Time,Pop/Rock,DSD,Restricted
- Sambasevam Shanmugam - Kaathaadi,Pop/Rock,DSD,Restricted
- Secret Mountains - High Horse,Pop,MedleyDB,CC BY-NC-SA
- Secretariat - Borderline,Pop/Rock,DSD,Restricted
- Secretariat - Over The Top,Pop/Rock,DSD,Restricted
- Side Effects Project - Sing With Me,Rap,DSD,Restricted
- Signe Jakobsen - What Have You Done To Me,Pop/Rock,DSD,Restricted
- Skelpolu - Human Mistakes,Electronic,DSD,Restricted
- Skelpolu - Resurrection,Electronic,DSD,Restricted
- Skelpolu - Together Alone,Electronic,DSD,Restricted
- Snowmine - Curfews,Pop,MedleyDB,CC BY-NC-SA
- Speak Softly - Broken Man,Pop/Rock,DSD,Restricted
- Speak Softly - Like Horses,Pop/Rock,DSD,Restricted
- Spike Mullings - Mike's Sulking,Pop/Rock,DSD,Restricted
- Spike Mullings - Mike's Sulking,Rock,DSD,CC BY-NC-SA
- St Vitus - Word Gets Around,Pop/Rock,DSD,Restricted
- Steven Clark - Bounty,Pop,MedleyDB,CC BY-NC-SA
- Strand Of Oaks - Spacestation,Pop,MedleyDB,CC BY-NC-SA
- Sweet Lights - You Let Me Down,Pop,MedleyDB,CC BY-NC-SA
- Swinging Steaks - Lost My Way,Pop/Rock,DSD,Restricted
- The Districts - Vermont,Rock,MedleyDB,CC BY-NC-SA
- The Doppler Shift - Atrophy,Pop/Rock,DSD,Restricted
- The Easton Ellises - Falcon 69,Pop/Rock,C't Remix Comp.,CC BY-NC-SA 3.0
- The Easton Ellises (Baumi) - SDRNR,Pop/Rock,C't Remix Comp.,CC BY-NC-SA 3.0
- The Long Wait - Back Home To Blue,Pop/Rock,DSD,Restricted
- The Long Wait - Dark Horses,Pop/Rock,DSD,Restricted
- The Mountaineering Club - Mallory,Pop/Rock,DSD,Restricted
- The Scarlet Brand - Les Fleurs Du Mal,Rock,MedleyDB,CC BY-NC-SA
- The So So Glos - Emergency,Rock,MedleyDB,CC BY-NC-SA
- The Sunshine Garcia Band - For I Am The Moon,Reggae,DSD,Restricted
- The Wrong'Uns - Rothko,Pop/Rock,DSD,Restricted
- Tim Taler - Stalker,Pop/Rock,DSD,Restricted
- Timboz - Pony,Heavy Metal,DSD,Restricted
- Titanium - Haunted Age,Heavy Metal,DSD,Restricted
- Tom McKenzie - Directions,Pop/Rock,DSD,Restricted
- Traffic Experiment - Once More (With Feeling),Pop/Rock,DSD,Restricted
- Traffic Experiment - Sirens,Pop/Rock,DSD,Restricted
- Triviul - Angelsaint,Pop/Rock,DSD,Restricted
- Triviul - Dorothy,Pop/Rock,DSD,Restricted
- Triviul feat. The Fiend - Widow,Pop/Rock,DSD,Restricted
- Voelund - Comfort Lives In Belief,Pop/Rock,DSD,Restricted
- Wall Of Death - Femme,Heavy Metal,DSD,Restricted
- We Fell From The Sky - Not You,Heavy Metal,DSD,Restricted
- Young Griffo - Blood To Bone,Heavy Metal,DSD,Restricted
- Young Griffo - Facade,Heavy Metal,DSD,Restricted
- Young Griffo - Pennies,Heavy Metal,DSD,Restricted
- Zeno - Signs,Pop/Rock,DSD,Restricted
