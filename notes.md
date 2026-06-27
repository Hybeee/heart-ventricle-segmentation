# Zajos CT-k
- 2: hangyafoci
- 13: kicsi hangyafoci
- 15: hangyafoci
- 21: kicsi hangyafoci
- 26: streaking artifact
- 34: hangyafoci + kevesbe kontrasztos
- 35: kicsi hangyafoci
- 38: kicsi hangyafoci
- 43: enyhe/sotet hangyafoci
- 45: streaking artifact
- 46: enyhe hangyafoci

## Median filter results (azok leirva, ahol a normálhoz képest változott az eredmény) - 3x3 filter
dT = (no_filter_threshold - with_filter_threshold)

- 3 -> kicsi valtozas, de mindketto jonak tunik dT = 13
- 5 -> nagyobb valtozas, kulso kis szigetet nem vett a szegmentacioba az adott szeleten. dT = -48
- 19 -> dT = -12, de nem latszik szinte semmi a kulonbsegben
- 26 -> ranezesre ugyanaz az eredmeny, dT = 12
- 31 -> megjavitotta a maszkot: dT = -33
- 32/33 -> dT = -17, de szinte ugyanaz a maszk
- 41 -> dT = -26, de szinte ugyanaz a maszk
- 43 -> dT = 14, mint elobb
- 48 -> dT = 109, es a median filteres egy fokkal jobb? bar az eredeti maszk se tunt rossznak
- 50 -> dT = 10, a maszkok kb. ugyanazok

Osszefoglalva:
- Javitott:
    - 31
    - 48: "javitott"
- Rontott:
    - 5: "rontott"
- Megjegyzes: Olyanoknal segitett, akik eredetileg nem voltak a zajos felvetelek reszei.

## Median filter results - 5x5
dT = (no_filter_threshold - with_filter_threshold)

- 6 -> dT = 24, a maszk kb. ugyanaz
- 8 -> dT = 18, a maszk kb. ugyanaz
- 10 -> dT = -18, a maszkban nincs nagy valtozas
- 13 -> dT = 20, a maszk javult? idk.
- 15 -> dT = 52, az eredeti nagyon kicsit alul, a filtered nagyon kicsit tulszegmental - az orvoshoz kepest
- 17 -> dT = 20, mindket maszk jonak nez ki
- 21 -> dT = -18, de mindket maszk kb. ugyanugy nez ki
- 27 -> dT = 10, a maszkok kozott nincs kulonbseg
- 31 -> dT = -30, hasonloan javit, mint a 3x3-as filter
- 32/33 -> dT = -17, de szinte ugyanaz a maszk
- 34 -> dT = -6, a maszk egyebkent kicsit javul? de nem jelentosen, plusz eleve a zajossag miatt mindketto rossz - eredeti, filterezett.
- 38 -> dT = 10, kb. ugyanaz a ket maszk
- 40 -> dT = -21, a maszkok egyebkent kb. ugyanazok, talan a filterezett jobb?
- 41 -> dT = -27 -> a maszkok kb. ugyanazok
- 48 -> dT = 106 -> ugyanaz, mint a 3x3-as filternel
- 50 -> dT = 10, a maszkok kb. ugyanazok
- 52 -> dT = 14, a maszkok kb. ugyanazok

Osszefoglalva:
- Javitott:
    - 13: talan? az orvosi is eleve rossz, plusz ez a special felvetel, ami eleve "zajkent" lett kuldve
    - 31
    - 40: mindket maszk jobb, talan a filterezett egy kicsivel jobb
    - 48: ugyanaz, mint 3x3-as esetben
- Rontott:
- Megjegyzes: Olyanoknal segitett, akik eredetileg nem voltak problemas felvetelek. Itt se segitett sokat a 31-est leszamitva - illetve a 13-ast, aki viszont gyakorlatilag kivetel.

## TV filter - bregman, weight=10.0
dT = (no_filter_threshold - with_filter_threshold)

- 6 -> dT = 31, a ket maszk hasonlo, a TV filtered-on kevesebb a lyuk, ami lehet egyebkent rossz? A korvonala mindkettonek szinte ugyanaz
- 7 -> dT = -12, (nagyon) enyhen alulszegmental a TV.
- 8 -> dT = 14, de a maszkok kb ugyanazok
- 10 -> dT = -20, a maszkok egyebkent majdnem ugyanazok
- 11 -> dT = 20, a TV-s maszk talan egy KICSIT jobb. De az eredeti se nagyon pontatlan
- 13 -> dT = -8, erdekesseg, hogy a TV-s maszk egy kicsit simabb
- 15 -> dT = 37, egy kicsit talan jobb a TV-s filter, az eredeti - az orvoshoz kepest - mintha enyhen alulszegmentalna
- 17 -> dT = 41, a TV kicsit tulszegmental, de cserebe korbeoleli az orvos szegmentaciojat
- 18 -> dT = -13, de a maszkok kb. ugyanazok
- 19 -> dT = 21, a maszkok kb. ugyanazok
- 20 -> dT = 9, a maszkok kb. ugyanazok
- 21 -> dT = -11, a maszkok kb. ugyanazok
- 22 -> dT = 25, a maszkok kb. ugyanazok
- 27 -> dT = 46, a maszkok kb. ugyanazok
- 28/29 (ugyanazok a felvetelek) -> dT = -5, itt ront a TV
- 34 -> dT = 34, itt nagyon ront! Erdemes lehet megnezni, ratapad az nnunet-re a jobb kamra fele
- 35 -> dT = 17, talan itt jobb a TV?
- 37 -> dT = -13, a maszkok ugyanazok
- 38 -> dT = -27, TV alulszegmental
- 39 -> dT = 16, itt szerintem javit a TV
- 40 -> dT = -17, maszkok kb. ugyanazok
- 42 -> dT = -15, nagyon alulszegmental a TV
- 44 -> dT = -12, a maszkok kb. ugyanazok
- 45 -> dT = 84, megoldja a streaking problemat!
- 48 -> dT = 128, kb ugyanaz, az eredeti talan minimalisan alulszegmental - az orvosi maszkhoz kepest
- 49 -> dT = 66, a TV tulszegmental
- 50 -> dT = -53, a TV alulszegmental
- 51 -> dT = 0.29, kicsit simabb a TV maszk

Osszefoglalva:
- Otlet, hogy miert tudott simabb lenni: A TV a polar resampling elott van alkalmazva -> ez kepes azon valley-ket eltuntetni/megvaltoztatni, amikbe amugy beleragad az eredeti megoldas es igy egy kicsit masabb maszkot eredmenyez? 
- Javitott:
    - 11
    - 15
    - 35
    - 39
    - 45: streaking problemat megoldja -> Miert? Mert a zajszures hatasara szinte mindegyik magas threshold, ami meg valid, raragad a streaking artifactra. Ami nem ragad ra, ott pedig altalanossagaban a volgyektol vett tavolsag sokkal nagyobb, mint a helyes megoldas eseten. Ez egyebkent nem feltetlen az a megoldas, amire szamitottam, de segitett.
    - 48: "javit"
- Rontott: 
    - 7
    - 17: ez lehet javitas/rontas is
    - 28/29: "rontas" - ugyanolyan rossz
    - 34: nnunet-re ratapad, egyik legdurvabb rontas
    - 38
    - 42
    - 49
    - 50

Measuring image noise - "Image Quality Assessment: From Error Visibility to
Structural Similarity": https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf

# Multi-slice algorithm
Érdekes esetek:
- patient_0004
    - Latszik, hogy az osszeesett maszk score-ja 2x akkora.
    - Viszont az osszeesett maszk elrontja a kimeneti maszkot is sajnos.
    - Egyebkent a tobbi szelet is kicsit fura. Lehet, hogy csak szerencsetlen valasztas
- patient_0005
    - Erdekes, hogy egy jobbnak latszo maszk score-ja rosszabb, mint egy rosszabbnak latszoe. Ez nem sulyos eset.
- patient_0006
    - Hasonlo, mint patient_0005
- patient_0012
    - Hasonlo, mint patient_0005, csak picit sulyosabb eset.
- patient_0014
    - Hasonlo, mint patient_0005
    - Kulon erdekes, hogy itt a legjobbnak latszo maszk score-ja lett sokkal magasabb
- patient_0017
    - Hasonlo, mint patient_0005
- patient_0019
    - Hasonlo, mint patient_0005, talan kicsit sulyosabb
- patient_0022
    - Hasonloan jo maszkok, de NAGYON eltero score ertekek
- patient_0023
    - Hasonlo, mint patient_0022
- patient_0027
    - Az egyik nagyon elter a masik kettotol
- patient_0028/patient_0029
    - Mindharom nagyon kulonbozo
- patient_0039
    - Egyik legfurabb eset, ahol a legjobbnak tuno maszk score-ja a legnagyobb/legrosszabb
- patient_0041
    - Mindharom maszk jo, de valamiert az egyiknek 2x annyi a score-ja
- patient_0044
    - Hasonloan az egyik maszk eseten a score nagyon kulonbozik/tulsagosan nagy. Fura felvetel, ugyhogy nehezebb a maszkok helyesseget is megallapitani
- patient_0048
    - Latszik, hogy legalacsonyabb score -> legszebb szelet -> stabilabb eredmeny.
    - Az eredmeny threshold ugyanaz, mint a filterek (median, TV) altal javitott esetekben
- patient_0050
    - Itt is erdekes, hogy a legjobbnak kinezo maszk score-ja a legnagyobb
- patient_0051
    - Segit az algoritmus


# Other
- patient_0020 streaking artifact, majd detector-ral le kell checkolni (nem detektalja rendesen)
- Streaking artifact-okhoz erdekesseg: tobb helyen megjelennek, de csak patient_0045 eseten okoznak gondot. Mas esetben - pl. patient_0048 - nem. Ennek a miertje a polar plot-on jol latszik; szimplan elkeruli az nnunet a streaking artifact-ot, a sugar es a maszk boundary metszesenel nincsen extrem gradiens ugras.
- Multi-slice algoritmushoz megjegyzes
    - Lehet erdemes lenne megnezni nem csak a maszkok minosege es a score-ok kozotti korrelaciot, hanem a threshold ertekek es a score-ok kozotti korrelaciot
    - Erdemes lehet kulonbozo megkozeliteseket megnezni a vegso threshold eloallitasara
        - sulyozott atlag (most)
        - minimum score-hoz tartozo threshold (ez szerintem bizonyos esetekben elrontja a megoldast)

# 3D masks
## patient_0001
mindharom maszk fele jo, de ugyanaz a problema elojon alul, illetve felul. Megallapithato, hogy a reconstr maszk a legjobb.

Megjegyzes: Maszk sokszor (relativ ertelemben) kicsit sokat valtozik ket szelett kozott, -260.6250mm-nel ket szelett kozott figyelheto meg. Valszeg a rekonstr alg.-ot kell finomitani.

alul:
![patient_0001_lower](notes_data/patient_0001_lower.png)

felul
![patient_0001_upper](notes_data/patient_0001_upper.png)

## patient_0002
alul:
normal maszk rossz - mindketto.
w_mean es w_med kb. hasonlo, jo eredmenyeket produkalnak. Viszont, minden esetben lyukacsos a maszk a zaj miatt -> dilatacio?
Egyik vegben meg kis pontok vannak a maszkban -> erozio/helyes reconstr alg megoldja

Mindharom esetben latszik, hogy kell meg a reconstr algot kicsit pofozni. Itt segitett a multislice alg.

Alja: (erdekesebb, tetejen csak a lyukacsossag figyelheto meg, egyebken u.a. mint 1-es esetben)
![patient_0002_lower](notes_data/patient_0002_lower.png)

## patient_0003
szinte tokeletesek a maszkok mindharom modszer eseteben, nem jon fel alul/felul problema. Latszik, hogy a reconstr alg. itt segit -> a sima/nyers maszk jelentosen rosszabb, vannak pl. "kinyulasai"

## patient_0004
mindharom maszk jo, de
- streaking artifact elrontja -> ezzel nem sok mindent lehet kezdeni
- az algoritmus (patient_0002-hoz hasonloan) felremegy, kell elotte egy dilatacio

streaking:
![patient_0004_streaking](notes_data/patient_0004_streaking.png)

alul, illetve felul hasonlo a jellemzo, mint 1-es eseteben. illetve lehet, hogy kellene pl erozio majd dilatacio amiatt, ami alul, illetve felul van:

alul
![patient_0004_lower](notes_data/patient_0004_lower.png)

felul
![patient_0004_upper](notes_data/patient_0004_upper.png)

Erdekesseg: a maszk lyukacsossaga - ez egyebkent tobb helyen fordul elo -, a kovetkezo 3D-s maszkot eredmenyezi:

![patient_0004_3d](notes_data/patient_0004_3d.png)

## patient_0005
streaking itt is problema:

![patient_0005_streaking](notes_data/patient_0005_streaking.png)

egyeshez hasonloan itt is van olyan, hogy a reconstr maszk valtozgat szomszedos szeletek kozott -> alg finomitasa kell

egyebkent mindharom maszk jo es alul/felul nincs problema

## patient_0006
a normal algoritmus maszkja kicsit 'darabos', de nem a dilatacio hianya miatt - emiatt egyebkent a reconstr maszk is szetmegy:

![patient_0006_norm](notes_data/patient_0006_norm.png)
![patient_0006_reconstr](notes_data/patient_0006_reconstr.png)

Mindharom algoritmusnak ez egyebkent problemat okoz. A felvetellel kb ez magyarazhato:

![patient_0006_ct](notes_data/patient_0006_ct.png)

Itt viszont alul/felul nincs az elsonel tett megfigyeles!

## patient_0007
alul egy kicsit problemas, de egyebkent a maszkok jok/ugyanaz a megfigyeles, mint a tobbinel:

![patient_0007_lower](notes_data/patient_0007_lower.png)

## patient_0008
Nincs alul/felul baj. A maszkok jol neznek ki. Egyedul a reconstr maszk 'villogasa' figyelheto meg, de az majd az alg. modositasaval kikuszobolheto lesz

## patient_0009
4-eshez hasonlo also, 1-eshez hasonlo felso, de egyebkent mindharom maszk jonak nez ki

## patient_0010
ranezesre mindharom maszk jo. Ami rossz, az mar az elozoekben megjelent. Itt is kellett volna dilatacio a kimeneti maszkon a reconstr alg. alkalmazasa elott.

## patient_0011
a maszkok kb ugyanazok. A dilatacio itt is segitene, illetve van egy fura resze:
![patient_0011_data](notes_data/patient_0011_data.png)

A rendes felvetelen kevesbe latszik a kontraszt, de lathatoan lefedi a magasabb kontrasztu reszt a maszk

## patient_0012
az alja/teteje mar latott.
a maszkok mindegyike jonak nez ki.
itt is latszik hogy a reconstr alg. finomhangolasa szukseges

## patient_0013
eleve zajos felvetel, de ugy nez ki mintha mindharom maszk egesz jol szegmentalna. Itt is kelleni fog dilatacio. Erdekes modon a rekonstrualt maszk nem esik szet, pedig az eredeti, kimeneti maszk lyukas.

## patient_0014
az eleje/vege mar olyan, ami elojott (pl. patient_0004/patient_0001). Egyebkent az egyetlen problema itt is az, hogy 'villog' a rekonstrualt maszk.

Egyebkent mindharom maszk jol szegmental

## patient_0015
mindharom maszk jo, az alja, illetve a teteje is jo, de:
kellene dilatacio, mert az eredeti maszk lyukacsos.

Az aljan, illetve a tetejen kellene erozio, mert impulzus zaj-szeru a maszk:
![patient_0015_erosion](notes_data/patient_0015_erosion.png)

## patient_0016
mindharom maszk jo!

a teteje hasonlo, mint az egyes esetnek, also resszel nincsen gond

itt is problemas a rekonstrualt maszk bizonyos helyeken -> lazan kotott, de szerintem hozza tartozo resz eltunik -> piros vonal = GT maszk
![patient_0016_loose](notes_data/patient_0016_loose.png)

## patient_0017
talan egy picit alulszegmental itt az algoritmus, de az aljan pl erdekes:

![patient_0017_interesting](notes_data/patient_0017_interesting.png)

megjegyzes: a CT kevesbe kontrasztos

## patient_0018
mindharom maszk jo. Latszik, hogy az algoritmus finomitasa kell, hogy a final mask jo legyen. Illetve felul itt is latszik, hogy szukseg lehet eroziora

## patient_0019
itt is kb ugyanaz allapithato meg. Illetve a kovetkezo kep miatt erdemes lehet ranezni rendesen:

![patient_0019_interesting](notes_data/patient_0019_interesting.png)

## patient_0020
Alapvetoen mindharom maszk jo. Itt is latszik, hogy szukseg lesz a reconstr. algoritmus finomhangolasa/atirasa.

Erdekesseg, hogy az nnU-Net rosszul szegmental. Ez kikuszobolheto a postprocessing algoritmussal:

![patient_0020_nnunet](notes_data/patient_0020_nnunet.png)

## patient_0021
mindharom maszk ugyanolyan

Alul impulzus zaj-szeruek a maszkok, bar ezt reszben kezeli a rekonstr alg.

Erdekes ezenkivul:
![patient_0021_data](notes_data/patient_0021_data.png)

## patient_0022
A maszkok alja hasonlo patient_0002-hoz

A maszkok egyebkent kb. ugyanolyanok, viszont itt is latszik, hogy finomitani kell az algoritmust!

## patient_0023
Erre erdemes lehet kulon ranezni, nehez megmondani, hogy eleve mi a jo maszk/az orvose jo-e

![patient_0023_interesting](notes_data/patient_0023_interesting.png)

## patient_0024
Mindharom maszk rendes, illetve reconstr. valtozata is teljesen jo. Egyedul a tetejen erdekes egy kicsit, de azt leszamitva jo!

![patient_0024_upper](notes_data/patient_0024_upper.png)

## patient_0026
Mindharom maszk mindket valtozata tokeletes. Erdekesseg, hogy itt latszik a streaking artifact minimalis hatasa - maszk alakja jo, de a sugar (streaking artifact szempontjabol) iranyaban van lyuk a maszkban:

![patient_0026_streaking](notes_data/patient_0026_streaking.png)

## patient_0027
jok a maszkok. Az aljak jok - foleg rendes postprocessing-el -, mig a teteje patient_0001-hez hasonlit.

## patient_0028
Alapvetoen rosszabb felvetel. Kozepen kb. jok a maszkok (a "weighted mean" modszer alulszegmental kicsit), viszont felul kicsit problemas a helyzet:

![patient_0028_problem](notes_data/patient_0028_problem.png)

## patient_0029
Duplikalt felvetel, ugyanaz, mint patient_0028

## patient_0030
Gyakorlatilag tokeletes felvetel mindharom maszk szempontjabol.

## patient_0031
Erosen latszik, hogy itt segit a multislice algoritmus. A multislice-os maszkok eseten mar szep az eredmeny. Talan a teteje erdekes lehet:

![patient_0031_upper](notes_data/patient_0031_upper.png)

## patient_0032
Mindharom maszk jo. A teteje patient_0001-re hasonlit. Az aljan eredekes, valszeg a reconstr. alg. javitasaval kikuszobolheto:

![patient_0032_lower](notes_data/patient_0032_lower.png)

## patient_0033
Ugyanaz, mint patient_0032 (nem pontosan, "mintha kisebb lenne a felbontas"). Megtalalt eredmenyek ertelmeben viszont ugyanaz, igy az elozo allitasok ervenyesek.

## patient_0034
Rosszabb minosegu felvetel, kb. impulzus zaj-szeru maszk -> a reconstr. maszk emiatt szetesik. Viszont rendes algoritmussal szerintem jol neznenek ki a maszkok (vagy pl. csak egy erozio majd dilatacio a reconstr. alg. elott). Emellett mintha a multislice alg.-ok kicsit jobb threshold-ot hataroznanak meg (285/286 vs. 297)

## patient_0035
Mindharom maszk ugyanolyan, viszont az impulzus zaj-szeru maszk hasonloan lathato itt is, mint az elozo felvetel eseten. Az eredeti/postprocessing elotti maszk latszolag jol hatarolja korbe az orvosi/GT maszkot (az egyeb zaj akorul egy gyenge erozioval pl. eltuntetheto lenne). ->  reconstr. alg. megvaltoztatasa kell.

## patient_0036
Alapvetoen mindharom maszk jo, de itt is erdekes, hogy mit lehet a kovetkezovel tenni:

![patient_0036_interesting](notes_data/patient_0036_interesting.png)

## patient_0037
Mindharom maszk tokeletes. Viszont itt is latszik, hogy javitani kell a reconstr. algoritmuson. A maszk bizonyos helyeken 'villog' - (ket szomszedos szelet eseten nagyobb regio benne van majd nincs majd megint benne van)

## patient_0038
Impulzus zaj-szeru maszkok, amik jol lefedik az orvosi/GT maszkot, de emiatt a rekonstr. maszk felremegy.

## patient_0039
ugyanaz figyelheto meg, mint elobb

## patient_0040
Alapvetoen mindharom maszk - reconstr.-okkal egyutt - jo, de erdekes lehet ez majd a finomhangolt algoritmus szamara:

![patient_0040_interesting](notes_data/patient_0040_interesting.png)

## patient_0041
Itt is jok a maszkok a stabil szeleteken, viszont van itt is erdekesseg:

![patient_0041_interesting](notes_data/patient_0041_interesting.png)
![patient_0041_interesting_raw](notes_data/patient_0041_interesting_raw.png)

## patient_0042
Alapvetoen ugyanaz allapithato meg, mint patient_0038-nal. Viszont pluszban, felul lehet, hogy a GT maszk rossz?

![patient_0042_badgt](notes_data/patient_0042_badgt.png)

## patient_0043
Zajos maszkok (impulzus) itt is. Viszont erdekes, hogy pontosan fedik az orvosi maszkot az nnU-Net-en belul:

![patient_0043_interesting](notes_data/patient_0043_interesting.png)

Emellett mintha kicsit tulszegmentalnanak az algoritmus altal eloallitott maszkok.

## patient_0044
Hasznalhatatlan a GT maszk sajnos. Egyebkent a maszkok nem tunnek rossznak.

## patient_0045
Itt van a nagyon durva streaking artifact. Erdekes, hogy az alap maszkok mindenhol jok. Viszont, mind az alap, mind a rekonstr. maszkja az eredeti algoritmusnak rosszabb -> sokkal lyukacsosabb a streaking miatt, ami elrontja a postprocesselt maszkot is. Ezzel szemben a multislice esetben olyan threshold-ot talal meg az algoritmus, ami ez ellen robusztusabb. A streaking itt kevesbe rontja el a maszkokat. Erdekes megfigyelni azt is, ahogy a streaking artifact vegigvonul az axialis szeleten es magaval huzza a maszkot.

## patient_0046
Zajos maszkok. Erdekes, hogy az eredeti tulszegmental(? legalabbis a jobb kamra iranyaba biztos), viszont a rekonstr. az gyakorlatilag tokeletes:

![patient_0046_interesting](notes_data/patient_0046_interesting.png)

## patient_0047
Itt is mindharom maszk gyakorlatilag tokeletes, viszont itt is megfigyelheto a reconstr. maszk "villogasa".

## patient_0048
A multislice maszkok tokeletesek, a normal nem a legjobb. Itt is van egy eros streaking artifact, ez rontja el a normal maszkot. Ahol nincsen streaking, ott a normal is nagyjabol jo - talan egy kicsit lyukas. A normal algoritmus eleve egyebkent sokkal magasabb threshold-ot talalt meg (288/290 vs 389)

## patient_0049
Lyukas maszk a normal alg. eseteben gondot okoz (reconst. maszknal foleg). A masik ket, multislice esetben a maszkok nagyon jok. A teteje kb. olyan, mint patient_0001 eseteben.

## patient_0050
Alapvetoen mindharom modszer jo, a normal es a weighted mean mintha kicsit alulszegmentalna, a weighted median jobb.

## patient_0051
A stabil szeleteken mindharom modszer jo, azt leszamitva, hogy a maszkok lyukacsosak, ezzel pedig a reconstr. maszk is szetmegy. Illetve erdekes, hogy alul, illetve felul a kovetkezo tortenik (zajossag):

![patient_0051_interesting](notes_data/patient_0051_interesting.png)

Megjegyzes: A felvetel eleve nem tul kontrasztos.

## patient_0052
Mindharom modszer nagyon jo. Viszont itt is latszik, hogy finomitani kell a reconstr. algoritmust; a postprocessing elotti maszk mindharom esetben nagyon jol lefedi a GT maszkot, viszont a postprocessing utan eloallo maszk sokszor kisebb lesz/reszek leszakadnak="villog"

## patient_0053
A GT maszk itt biztosan rossz. Mindharom modszer maszkja tokeletes, viszont itt is megfigyelheto, hogy szomszedos szeleteken a reconstr. maszk "villog" -> finomhangolni kell az algoritmust

# 3D masks - 3D reconstruction algorithm
- patient_0006-nal tovabbra is alulszegmental a megoldas
- patient_0007 -> talan feljebb kene venni az egyik kuszobot, de amugy jo
- patient_0010 -> kicsit alulszegmental, szivizom miatt lehet ez foleg erdekes?
- patient_0011 -> feljebb kell kicsit venni a hidas threshold-ot, itt is van egy kicsi leak
- patient_0013 -> erdekes, de lehet megoldja az impulzuszajbol fakado problemat?
- patient_0019 -> az eredeti megoldas se a legjobb, igy a reconstr se. De amiert lett irva, azt jol elvegzi itt is!
- patient_0023 -> erdekes, hogy egy nagyon nagyon kicsi leakage, ami nagyon vekony uttal van osszekotve nem tunt el az algoritmus hatasara?
- patient_0028/patient_0029 -> ezek is szebbek lettek!
- patient_0031 -> tulszegmentalas talan? nehez eldonteni
- patient_0034 -> alapvetoen is rossz a zajossag miatt az eredmeny, de a reconstr alg. javit a maszkon!
- patient_0036 -> talan kicsit tulszegmental
- patient_0042 -> valamiert itt is kicsit felremegy az algoritmus, majd meg kell nezni, hogy miert. Otlet: erozio utan largest connected component megtartasa es abbol dilatalas. A 3D-s volume-ot nezve ott mehet felre. Plusz lehet az n_erosions_needed rosszul szamolodik ki valamiert?
- patient_0046 erdekes lehet! Az alap GT maszk is rossz/lyukacsos, emiatt itt nehezebb kiertekelni, bar szerintem jo a vegso maszk.
- patient_0051 -> itt is elojon, hogy kicsit magasabb threshold kellene a vastagsagot illetoen

# kovi konzira erdekesek:
- patient_0007: erdekes leak? miert nem szedi ki? todo otlet: tul kicsi leak -> nem lesz belole island
    - erdekes az imfilled maszk
    - nem detektal island-et. Szerintem azert, mert nem eleg vastag a leak?
- patient_0011: van leak, szerintem itt a vastagsag miatt nem vagta le
    - ugyanaz a problema, mint 0007-nel
- patient_0019: alulszegmentalas
- patient_0023: mini leakage, talan ugyanazon okbol, mint 0007?
    - mint 0007-nel
- patient_0026: itt is van mini leak, de eltunik. erdemes osszehasonlitani az algoritmus mukodeset pl. 23-mal.
    - nem tunik vastagnak a leak, megis eltunik es rendesen detektal szigeteket az alg.. ugyanakkor erdekesen nez ki az island mask, mintha a korvonala lenne a maszknak
- patient_0031: itt is van egy kicsi leak, de ezt ranezesre a threshold novelese megoldja
    - megoldja a threshold novelese
    - de az island mask sum 1...
- patient_0041: hasonloan, mint patient_0026-nal
    - 2.22-vel jo, de 3.5-tel mar tul sokat vag le.
- patient_0045: van-e erozio? szerintem nem, de a ket maszk meg kicsiket mas
    - Megoldas: Nagyobb az erozio, mint 0. Szoval van (helyesen) -> mas a ket maszk.
- patient_0051: itt is van egy kicsi leak es ezt is valoszinuleg a threshold novelese megoldja majd
- patient_0053: ugyanaz, mint patient_0045-nel
    - ugyanaz a helyzet -> igazabol van minimalis leak valahol

# island detection with marching_state[background] = 1000
- patient_0001: ok
- patient_0002: ok
    - Megjegyzes: ugyanaz, mint patient_0001
- patient_0003: ok
    - Megjegyzes: n_erosions = 1, de a visszadilatalas itt nem rontja el, szep a maszk.
- patient_0004: teljesen jo
    - Megjegyzes: streaking artifact itt beleront a maszkokba, de ez pl a GT-n is latszik. Szelettol fuggoen a GT/prediktalt maszk robusztusabb.
- patient_0005: ok
- patient_0006: talan itt megy egyedul felre a javito algoritmus
    - Megjegyzes: nem az island detektalasok hibaja igazabol, bar TODO hogy miert.
- patient_0007: ok
- patient_0008: ok
    - Megjegyzes: a base fele van egy-ket szelet, ahol a patient_0006-hoz hasonloan eltavolitodik a maszk egy resze, de a 3D-s modell pedig jo
- patient_0009: ok
- patient_0010: alapvetoen szerintem itt a felveteje hibaja miatt van baj. A szivizmok reszei (?) 3D-s modellen nyitottak.
- patient_0011: alapvetoen egy MININIMALIS leak van, az valamiert nem kerul le, le kell checkolni TODO
    - Megjegyzes: ezenkivul ok, de az aljanal az alap maszk is alulszegmental (? szerintem nem, gt maszk rossz)
- patient_0012: ok
- patient_0013: zajos CT, kimegy a masik oldali kamraig. Egyebkent nem rossz az eredmeny - a masik kamraig valo kinyulast leszamitva.
- patient_0014: ok
- patient_0015: zajos CT (~patient_0013), de amugy ok
- patient_0016: ok
    - Megjegyzes: ~1 voxellel alulszegmental 
- patient_0017: ok
- patient_0018: ok
- patient_0019: eredeti felvetel meh, fura a kontrasztos resz, emiatt az algo elhasal
    - Megjegyzes: nem a postproc hibaja
- patient_0020: alapvetoen ok
    - Megjegyzes: vannak hasonlo reszek, mint patient_0006-nal
- patient_0021: zajos CT, mindharom megoldas rossz.
    - Megjegyzes: javit valamennyit a postproc alg.
- patient_0022: ok, de egyebkent eleve rossz az orvos? mintha a jobb kamra lenne kontrasztos?
    - Megjegyzes: enyhen alulszegmentalja mindket maszk az orvoset.
- patient_0023: rossz kontraszt miatt alulszegmentalas?
    - Megjegyzes: alg jo.
- patient_0024: ok
    - Megjegyzes: elojon a szivizmos dolog, illetve itt is egy kicsit a hidba beleszegmentalas tortenik. de egyaltalan nem rossz az eredmeny 3D modell! -> szivizmos dolog = patient_0006-nal -> bizonyos kinovesei az eredeti maszknak lemetszodnek -> dilatacio mar nem allitja vissza.
- patient_0026: ok
- patient_0027: ok
- patient_0028: ok
    - Megjegyzes: itt nincs leak, csak egy kicsi, viszont az ahhoz tartozo hid picit itt is benne van
- patient_0029: ekvivalens patient_0028-cal
- patient_0030: ok
- patient_0031: itt elbukik. szerintem tul vastag a leak. 3D-s modellnel rosszabbul nez ki a hiba, mint axialis szeletek eseten -> ellenorzes TODO
- patient_0032: ok
- patient_0033: ekvivalens patient_0032-vel
- patient_0034: mindharom maszk rossz a felvetel minosege miatt
    - Megjegyzes: hidas dolog itt is problemas!
- patient_0035: ok - zajos felvetel, de jo szerintem eleve is a (vegso) maszk
- patient_0036: zajos az eredeti kimenet, nem tudom eldonteni, hogy ott egy leak van-e vagy sem
    - Megjegyzes: viszont a postproc felnagyitja azt a reszt, de a 3D-s modell nem nez ki rosszul.
- patient_0037: ok
- patient_0038: ok
    - Megjegyzes: zajos kimeneti maszk
- patient_0039: ok
- patient_0040: ok
- patient_0041: ok
- patient_0042: tul vastag hid az eredetiben, szerintem az a baj de TODO check
- patient_0043: zajos felvetel, zajos orvosi es base maszk, de az utofeldolgozott eredmeny jonak tunik, talan tulsagosan kinyulik a masik kamra fele?
- patient_0044: rossz gt maszk, de mukodik az algo
- patient_0045: ok
- patient_0046: ugyanaz, mint patient_0043, vagy akar patient_0013
- patient_0047: mini alulszegmentalas? algo jo
- patient_0048: ok
- patient_0049: ok
- patient_0050: kicsi alulszegmentalas, de algo jo, illetve inkabb szerintem az gt maszk szegmental tul
- patient_0051: ok
- patient_0052: ok
- patient_0053: ok
    - Megjegyzes: orvosi maszk alulszegmental

## TODO-k
- Hid felnagyitasa
    - Megfigyeles: a mostani algoritmus felnagyitja a hidak/leak-ek egy reszet. 
    - Kerdes: A megirt algoritmus, ami az eredmeny maszkot a 'non-increasing path' modon allitja elo mennyire produkal mas maszkokat (ott is megfigyelheto volt ez a jelenseg.)
    - Eredmenyek:
        - valamiert - meg mindig - bennemaradt egy fontos resz a kodban: az elejen egy dilatacio, ami miatt a hidak vastagodnak es igy benne maradnak. le kell ujra tesztelni, de ez elmeleti szinten megoldja az osszes ilyen problemat.

# Potencialis javitas; elejen dilatacio kivetele, illetve marching_state[background] = large_value es `non_increasing_path` modszer osszevetese | orig vs. nip
- patient_0001: ok, orig
- patient_0002: egyik sem jo. Rossz alapmaszk, tul aggresszivan lecsip belole az algoritmus -> a vegsok alulszegmentalnak
    - Megjegyzes: Itt pl. segitett a binary closing. TODO: Megnezni, hogy a lyukakat betomkodo maszk kimenete mi. Osszevetve akar a closing-os maszkkal.
- patient_0003: ok, orig/nip
- patient_0004: o volt a fo motivacioja a closing-nak.

# Vegso background vs non increasing path osszevetes
- patient_0001: 
    - Maszk: ok
    - Egyeb: mindket alg ok. nip kicsit tobbet hagy meg a hidbol, de o is ok. Szeleteket tekintve minimalis kulonbseg
- patient_0002: 
    - Maszk: ok, zajosabb eredmeny mar a GT maszknal is.
    - Egyeb: Szeleteket tekintve nagyreszt ekvivalens a ket maszk, base fele van resz, ahol nip egy kinyulast kicsit alulszegmental.
- patient_0003: 
    - Maszk: Ok, minden esetben
    - Egyeb: Talan itt nip a jobb, illetve szivizmok megjelenesenel (amikor meg a szelen vannak egy szeletet tekintve) jobbnak tunik - az ott levo kis 2D-s hid nem tunik el.
- patient_0004: 
    - Maszk: mindegyik maszk ok
    - Egyeb: Talan zajosabb egy kicsit a szivizom szele - szeletet tekintve - nip eseten. De kb. ekvivalens a ket megoldas. Illetve nip jobban fennragad a streaking-re, bar ez mindkettore igaz
- patient_0005: 
    - Maszk: ok
    - Egyeb: max a hidas dolog emlitheto itt is
- patient_0006: 
    - Maszk: alapvetoen a CT kontrasztja rosszabb, igy az eredmenymaszkok nem tokeletesek
    - Egyeb: alg-ok futasa okes, neha nip neha background szegmental jobban/alul
- patient_0007: 
    - Maszk: alapvetoen ok
    - Egyeb: mindket alg betomi a szivizmot. Lehet ez az uj modszer miatt van? fura.
- patient_0008: 
    - Maszk: ok
    - Egyeb: -
- patient_0009: 
    - Maszk: ok
    - Egyeb: -
- patient_0010: 
    - Maszk: ok
    - Egyeb: szivizom hid + maszk szele
- patient_0011: 
    - Maszk: ok
    - Egyeb: -
- patient_0012: 
    - Maszk: ok
    - Egyeb: -
- patient_0013: 
    - Maszk: ok
    - Egyeb: nagyon zajos alapfelvetel es kimeneti maszkok impulzus-zajszeruek. Ezt javitja a postproc alg
- patient_0014: 
    - Maszk: ok
    - Egyeb: -
- patient_0015: 
    - Maszk: ok
    - Egyeb: ~patient_0013
- patient_0016: 
    - Maszk: ok
    - Egyeb: szivizmos dolog
- patient_0017: 
    - Maszk: ok
    - Egyeb: -
- patient_0018: 
    - Maszk: ok
    - Egyeb: -
- patient_0019: 
    - Maszk: alulszegmental mar az eredeti maszk is
    - Egyeb: alg ok
- patient_0020: 
    - Maszk: ok
    - Egyeb: nip erzekenyebb/tobb bennemarad a streakingbol
- patient_0021: 
    - Maszk: ok
    - Egyeb: rossz CT eleve, mindegyik maszk alulszegmental (gt is)
- patient_0022: 
    - Maszk: ok
    - Egyeb: ~patient_0021
- patient_0023:
    - Maszk: ok
    - Egyeb: ~patient_0021
- patient_0024: 
    - Maszk: ok
    - Egyeb: -
- patient_0026: 
    - Maszk: ok
    - Egyeb: -
- patient_0027: 
    - Maszk: ok
    - Egyeb: szivizom dolog
- patient_0028: 
    - Maszk: ok
    - Egyeb: eleve rossz felvetel
- patient_0029: 
    - Maszk: patient_0028-cal ekvivalens felvetel
    - Egyeb: -
- patient_0030: 
    - Maszk: ok
    - Egyeb: -
- patient_0031: 
    - Maszk: ok, gt szerintem itt rosszabb
    - Egyeb: a felvetel maga is alacsony kontrasztu
- patient_0032: 
    - Maszk: ok
    - Egyeb: -
- patient_0033: 
    - Maszk: patient_0032-vel ekvivalens felvetel
    - Egyeb: -
- patient_0034: 
    - Maszk: ok
    - Egyeb: ~patient_0013, eleve alulszegmental mindegyik a felvetel minosege miatt
- patient_0035: 
    - Maszk: ok
    - Egyeb: ~patient_0013
- patient_0036: 
    - Maszk: ok
    - Egyeb: alacsony kontrasztu felvetel (emiatt alulszegmental mindegyik megoldas). gt rosszabb itt.
- patient_0037: 
    - Maszk: ok
    - Egyeb: -
- patient_0038: 
    - Maszk: ok
    - Egyeb: ~patient_0002
- patient_0039: 
    - Maszk: ok
    - Egyeb: kicsit alulszegmental mindegyik a gt-hez kepest
- patient_0040: 
    - Maszk: ok
    - Egyeb: -
- patient_0041: 
    - Maszk: ok
    - Egyeb: erdekes eredeti felvetel (furcsa kontraszt), ez kihat a megoldasra, de az alg. alapvetoen ok
- patient_0042: 
    - Maszk: ok
    - Egyeb: itt tul vastag a hid a leak-be
- patient_0043: 
    - Maszk: ok
    - Egyeb: ~patient_0013
- patient_0044: 
    - Maszk: ok
    - Egyeb: rosszabb CT ez is, de az alg-ok jol mukodnek
- patient_0045: 
    - Maszk: ok
    - Egyeb: talan itt is kicsit tobb marad a streakingbol a nip-nel
- patient_0046: 
    - Maszk: ok
    - Egyeb: ~patient_0013. Illetve itt "sokkal" masabb nip, nem tudom, hogy jo vagy rossz ertelemben
- patient_0047: 
    - Maszk: ok
    - Egyeb: ~1 voxel vastagsaban van alulszegmentalas
- patient_0048: 
    - Maszk: ok
    - Egyeb: -
- patient_0049: 
    - Maszk: ok
    - Egyeb: nem tudom, hogy itt a CT felvetel minosege mindegyik alulszegmental-e?
- patient_0050: 
    - Maszk: ok
    - Egyeb: kisebb, de az eddigiekhez kepest, ahol "kisebb" nagyobb merteku alulszegmentalas.
- patient_0051: 
    - Maszk: ok
    - Egyeb: ~patient_0013 (sokkal kisebb mertekben)
- patient_0052: 
    - Maszk: ok
    - Egyeb: CT felvetel minosege miatt nem tudom, hogy mennyire szegmentalnak alul
- patient_0053: 
    - Maszk: ok
    - Egyeb: gt maszk biztos rossz

Pentekre fontos dolgok:
- hogy adhato el ezt lenne fontos atbeszelni
    - milyen problemakhoz hasznos, ehhez meg mi kellene
    - publikalhatosag szempontjabol (orvosi folyoiratok) mennyira adhato el igy
    - doktori felvetelhez kellenenek meg publikaciok (SOTE ok de egyebkent is jol jon)
- Mit mutassak:
    - Jo felvetelek:
        - patient_0001 - van leak!
        - patient_0005
        - patient_0008
        - patient_0045 - itt van streaking, de jol turi az algoritmu
        - patient_0053
    - Szivizmos/hidas dolog:
        - patient_0003 - itt inkabb az az erdekes, hogy miert van tobb "lyuk" a maszkban
        - patient_0005 (o egyebkent jo!!!!) - 1649.6999mm, 1632.8999mm
        - patient_0010 - -156.1500mm, egyebkent itt szerintem az enyem jo
    - zajos CT-s:
        - patient_0002 - van leak
        - patient_0021
        - patient_0038 - o jobb, mint patient_0002 eredmeny szempontjabol
    - rosszabb/kerdojeles eredmeny:
        - patient_0006 - alacsonyabb/furabb kontrasztu felvetel
        - patient_0007 - ha feljonne, akkor itt betomodik az alg. miatt a sziv. de itt van leak!! ok: z axis menten kicsi a felvetel -> nincs olyan 3x3x3 block a szivizomnal, ahol legalabb egy voxel ne lenne - helyesen - threshold-ot maszk resze -> closing bezarja, erozio ezt pedig nem oldja meg. Egyebkent az elofeldolgozast/lyukak bezarasat tekintve a felvetel "kicsisege" nem tunik elsore problemasnak. Lehet mas esetben az lenne, de itt nincsenek lyukak/alagutak, illetve ha lennenek is, akkor is betomodne a legtobb.
        - patient_0019 - mar az eredeti maszk is kicsit alulszegmental szerintem, az algoritmus foleg
        - patient_0036 - alacsony kontraszt, szerintem a gt maszk rosszabb itt
        - patient_0049 - a gt-hez viszonyitva jo, de lehet alulszegmental mindenki

- rolling ball-os konvex burkolo

- szivizmos dolgok patient_0003

Kovi hetre megnezni:
- papillaris szivizmok szegmentacioja mennyire van meg
    - mennyire pontos
- streaking artifact-os CT-kkel mi a helyzet
    - tobb polarkoordinatas streaking-es 2D-s kep kellene ott nezni hogy van-e valami

# Streaking artifcat-os felvetelek
- patient_0005(!)
    - streak-ek lyukat okoznak a maszkban
    - palmafaszeru mintazat a gradiensterkepen
    - a lyukak nem szivizomnak a reszei
    - alapveto problema: a maszk csak a legkulsobb radialis pontot jeleniti meg -> ezt konnyu valtoztatni
- patient_0008
    - itt nem szolnak bele annyira a streak-ek
    - kevesbe latszik a palmafas mintazat, a kamra volgye megmarad rendesen
- patient_0020
    - mintazat itt nincsen, a grad. map hasonlit arra, mint egy normal szeletnel
    - viszont itt megfigyelheto a ratapadas a streaking-re.
        - ez szerintem javithato interpolacioval
- patient_0026
    - nem annyira nagy a streaking, nincs ratapadas, inkabb csak a palmafas mintazat figyelheto meg
    - postprocessing leak-nek erzekeli a streak-hez valo kinyulast -> levagja
- patient_0030
    - enyhen megfigyelheto palmafa mintazat
    - nem okoz lyukakat
    - nincs ratapadas (nip ratapad)
- patient_0033 (minimalis streaking szerintem nem is latszodna)
- patient_0037
    - kicsi streaking
    - minimalisan - bar az lehet eleve a grad map - latszik a palmafa pattern
    - 124-es szeletnel mintha egy kicsit az szolna bele a maszk szelebe, de szerintem nem
    - alap maszk ratapad, de a post processing maszk leszedi
- patient_0039
    - nagyon kicsi streaking, illetve nem er bele a kamraba
- patient_0045
    - eszlelheto enyhe ratapadas a streaking-re.
    - nincs palmafa mintazat (minimalisan van max)
    - postprocessing ugy nez ki, hogy tudja ellensulyozni a streaking hatasat (lyukak/csikok a maszkban)
- patient_0048
    - ratapad a streaking-re es (Descartes-ban nezve) csikok lesznek a maszkban
    - elotte levo szeleteken kicsit kintebb nyulik a maszk az egyik sugar iranyaba, de ez minimalis.
- patient_0052: nem olyan streaking, mint ami kellene. kozepen van a bal kamranak es nincsenek "sugarak"

# Szivizom szegmentaciok vizsgalata
NOTE:
- koronalis szeleten talan atlathatobb/nem annyira furak az izmok
- axialis szeleteken erdekesebb, ezeket irtam le
- egyelore ugy nez ki, mintha thresholodolas tortenne, de kicsit mashogy - pl.: mellette levo, threshold feletti voxel nem a maszk resze? -> illetve van olyan, hogy a threshold szerint benne kene lennie, de nincs. lehet ezt irtam le elobb is, de igy tisztabb. pl 240 a maszk resze, de egy 232-es erteku voxel nem.
- valahogy ossze vannak a maszkok hangolva, ritkan van kozos voxele a kamranak es a szivizomnak
- az otlet, hogy kamra maszkban levo lyuk a szivizom biztos rossz 

Felvetelek:
- patient_0001: kb osszhangban a 3 maszk (LV, myo, izom). Kuszoboltnek nez ki a maszk. A kozepso szelet fele kb. szabad szemmel is lathato, hogy mi miert lett szegmentalva, feljebb/lejjebb viszont kerdojeles. szelen is vannak. Jobban latszik talan koronalis nezetbol?
- patient_0002: nem minden lyuk izom, de a legnagyobbak igen. Emellett mindket izom jelen van a maszk szelen (AL jobban). Keves lyuk van, es az is csak par szeletig lyuk, ami nem izom. Myocardium tulszegmental, illetve valahogy ugy van megcsinalva, hogy ne legyen atfedese a bal kamra szegmentaciojaval - NOTE: ennek neve heart_contour, nincs myocardium kulon. De a heart contour nez ki itt myocardiumnak. thresholding
- patient_0003: nem minden lyuk tartozik a szivizmokhoz es itt van tobb nagyobb is. CT szeleiig kimennek. nincs myocardium. Koronalis nezetbol itt is jobban latszik, illetve PM erdekesebb/lukacsosabb?
- patient_0004: szinte biztos hogy rossz a szegmentacio. kevesebb lyuk, de itt is igaz, hogy nem minden lyuk szivizom
- patient_0005: AL jonak tunik PM erdekes/alulszegmental? Kint vannak a szelen
- patient_0006: jonak tunik itt mindketto. A CT maga erdekes ("csikos" a kamra)
- patient_0007: itt is erdekesebb a szegmentacio, kint van a szelen
- patient_0008: PM kicsit erdekesebb. egyszerre nez ki threshold-olasnak meg nem is..
- patient_0009: PM erdekesebb
- patient_0010: itt is rossz lehet? AL nincs a kamraszegmentacio szelen/benne van teljesen, de szerintem azert, mert rossz az AL szegmentacio
- patient_0011: nem tudom eldonteni, hogy jo-e, szerintem nem. PM rosszabbnak tunik
- patient_0012: AL jo, PM rossznak tunik/nem ertem, hogy szabad szemmel hogyan lehetne meghatarozni
- patient_0013: teljesen zajos CT
- patient_0014: PM mogotti logikat itt sem ertem. mintha tulszegmentalna?
- patient_0015: nagyon zajos CT, de joknak/jobbnak tunnek, foleg PM
- patient_0016: thresholdos alapjan jonak tunik
- patient_0017: jonak tunik
- patient_0018: ez is jonak tunik
- patient_0019: base fele PM erdekes - axialis es koronalis szeleteken is. Hasonloan AL is erdekes
- patient_0020: jonak tunik/kevesbe fura
- patient_0021: nagyon zajos CT, PM kicsit rosszabbnak/erdekesebbnek tunik. AL megint olyan mintha tulszegmentalna?
- patient_0022: PM erdekesebb, de a threshold-os gondolatmenet alapjan okesak
- patient_0023: ugyanaz; PM erdekesebb.
- patient_0024: base-nel erdekes PM. egyebkent threshold-os gondolatmenet itt is allja a helyet. itt szebbnek neznek ki a maszkok a PM-es base-t leszamitva
- patient_0026: base fele PM erdekes, egyebkent latszatra is okes
- patient_0027: ugyanez
- patient_0028: rosszabb minoseguek a maszkok
- patient_0029: u.a., mint patient_0028
- patient_0030: itt lathatoan/erthetoen jo
- patient_0031: erdekesebbek itt is - mindketto
- patient_0032: mindketto erdekesebb
- patient_0033: nagyon a base-nel illetve nagyon az apex-nel mindketto erdekes
- patient_0034: nagyon zajos CT (~patient_0013), igy a maszkok kb. biztos rosszak/pontatlanok
- patient_0035: nagyon zajos CT, latszik hogy a maszkok is rosszabb minoseguek. Itt inkabb igaznak erzodik a "kamran beluli lyukak = szivizom".
- patient_0036: minimalisan latszodik a szivizmoknal levo kisebb intenzitas. Emellett PM kicsit furabb. FONTOS: itt pl. letezik olyan axialis szelet, amelyen mindket izom jelen van.
- patient_0037: itt jonak tunnek + magyarazhatonak is
- patient_0038: base fele PM megint erdekes, de zajos CT miatt itt sokkal kevesbe latszodik barmi.
- patient_0039: kevesbe, de valamennyire lathato. kiveve base fele PM
- patient_0040: apex es base resznel erdekes. PM vegig erdekesebb, kiveve ahol base fele eloszor megjelenik, AL magyarazhatobb
- patient_0041: Mindketto okesnak/magyarazhatonak tunik itt
- patient_0042: AL lathato a CT-n is, PM mintha "random" lenne
- patient_0043: ~patient_0013 -> valoszinuleg pontatlan szegmentaciok
- patient_0044: nincs osszhangban a kamra- es izomszegmentacio. Talan PM magyarazhato itt jobban?
- patient_0045: base fele PM erdekes, egyebkent, foleg AL, magyarazhatonak tunik
- patient_0046: ~patient_0013, pontatlan maszkok, de elnezve a CT-t kb. jo reszt jelolnek ki
- patient_0047: jonak tunik
- patient_0048: jol lathato es magyarazhato maszkok
- patient_0049: AL jobban, PM kevesbe magyarazhato, de eleve rosszabb a CT
- patient_0050: PM base fele erdekesebb, de amugy jol lathatoak az izmok a CT-n
- patient_0051: rosszabb CT/kevesbe lathato. Emiatt pontatlannak tuno maszkok
- patient_0052: AL kevesbe ertelmezheto, illetve a streaking kicsit elrontja PM maszkot
- patient_0053: eredeti LV maszk is el van rontva itt, hasonloan az izmok maszkjai
- patient_0054: rosszabb minosegu CT, de kb. helyesnek tuno maszkok.

sziv izok:
    - konvex burkolo(itt a zaj miatt nem olyan jo)/rolling ball a burkolora

streaking:
    - magja mennyire szegmentalhato? thresholding
    - ha igen akkor innen polarkoordinatas plot, mint center

# use-case-ek az algoritmusra
- Zhang et al., "Calculation of left ventricular ejection fraction using an 8-layer residual U-Net…versus echocardiography", Quantitative Imaging in Medicine and Surgery, 2023
    - https://qims.amegroups.org/article/view/116505/html
    - Cardiac CT angiography (CCTA) felveteleket hasznalnak
    - LVEF value a motivacio: left ventricular ejection fraction
        - szamos medical condition ezen value alapjan allithato meg, illetve pl. mutetekhez is hasznalt.
        - ehhez ha jol ertem az kell, hogy a kamra egyszer verre teli, egyszer meg ne verrel teli legyen.
- Zreik et al., "Automatic Segmentation of the Left Ventricle in Cardiac CT Angiography Using Convolutional Neural Network" (arXiv:1704.05698)
    - https://pmc.ncbi.nlm.nih.gov/articles/PMC7231613/
    - "Currently, cardiac CT is widely used to detect coronary artery and valvular heart diseases, as well as assess LV wall thickness and cardiac function."
    - myocardium szegmentacio itt a fo cel, a fenti idezet emlit kicsit a use-case-rol
- "Semantic Segmentation for Preoperative Planning in Transcatheter Aortic Valve Replacement" (arXiv, 2025)
    - Transcatheter Aortic Valve Replacement surgery-hez kell LV segmentation
- Mao, Y., Zhu, G., Yang, T., Lange, R., Noterdaeme, T., Ma, C., & Yang, J. (2024). Rapid segmentation of computed tomography angiography images of the aortic valve: the efficacy and clinical value of a deep learning algorithm. Frontiers in Bioengineering and Biotechnology, 12, 1285166.
    - ugyanugy TAVR-rol szol, nem emliti explicit, hogy ehhez kell LV szegmentacio, de az implementalt modellben kiertekeli az LV szegmentacio pontossagat
- Kim et al., "Left ventricular myocardium segmentation on arterial phase of multi-detector row computed tomography", Computerized Medical Imaging and Graphics, 2011
    - ez inkabb a myocardium szegmentaciorol szol, ami elvegezheto a mostani megoldassal (elobb szivizom szegmentacio, majd nnunet - (LV + szivizom) = myocardium)
- Chen, Z., Rigolli, M., Vigneault, D. M., Kligerman, S., Hahn, L., Narezkina, A., ... & Contijoch, F. (2021). Automated cardiac volume assessment and cardiac long-and short-axis imaging plane prediction from electrocardiogram-gated computed tomography volumes enabled by deep learning. European Heart Journal-Digital Health, 2(2), 311-322.
    - "Accurate and reproducible morphofunctional assessment of the left ventricle (LV) is crucial as LV morphology, volumes, ejection fraction (EF), and regional function are critical parameters used in the diagnosis,1 clinical management, prognostication, and follow-up of numerous cardiovascular and systemic diseases.2,3 The assessment of LV parameters is included in clinical guidelines1–3 and is used for both inclusion criteria and endpoints in clinical trials.4 In addition, regional LV wall motion abnormalities for 17 American Heart Association (AHA) LV segments are assessed using standardized views and are important for the evaluation of cardiac pathology, including coronary artery disease (CAD).5,6 Beyond the LV, the assessment of the left atrium (LA) provides additional insight into cardiovascular disease and function and is particularly important in evaluating patients with atrial fibrillation, valvular disease, and diastolic heart failure.7"
    - ir peldat, hogy mire jo a szegmentacio (fent). A fo celja viszont a cikknek egy olyan megoldas implementalasa, ami egyszerre szegmental es meghatarozza a SAX es LAX imaging sikok vektorait.
- Haq, R., Hotca, A., Apte, A., Rimner, A., Deasy, J. O., & Thor, M. (2020). Cardio-pulmonary substructure segmentation of radiotherapy computed tomography images using convolutional neural networks for clinical outcomes analysis. Physics and imaging in radiation oncology, 14, 61-66.
    - LV szegmentacio fontos sugarterapia eseten (cardiac dose constraints)
    - Eddig csak sziv + tudo szegmentaciora fokuszaltak, de egyre inkabb szukseg van a reszletesebb szegmentalasa ezen szerveknek
- Duane, F., Aznar, M. C., Bartlett, F., Cutter, D. J., Darby, S. C., Jagsi, R., ... & Taylor, C. W. (2017). A cardiac contouring atlas for radiotherapy. Radiotherapy and Oncology, 122(3), 416-422.
    - kb. ugyanaz a lenyeg, mint az elozonel
- Jung, J. W., Mille, M. M., Ky, B., Kenworthy, W., Lee, C., Yeom, Y. S., ... & RadComp Consortium. (2021). Application of an automatic segmentation method for evaluating cardiac structure doses received by breast radiotherapy patients. Physics and imaging in radiation oncology, 19, 138-144.
    - itt is az a lenyeg, hogy megnezzek, hogy mennyi sugarzas eri az LV-t/sziv reszeit -> kell a szegmentacio

# heart muscle segmentation initial alg eval
- patient_0001: kb. jol kijeloli a ROI-t. egyaltalan nem tokeletes
- patient_0002: outlier resze a maszknak elrontja a convex hull-t, emiatt elromlik az algo. Egyebkent igy is latszik a kb. jo irany
- patient_0003: outlier miatt elromlik
- patient_0004: itt is ez a baj, plusz itt alapbol rosszabbnak tunnek az eredmenyek
- patient_0005: plusz alakpriorral ez jol nezne ki, nem megy felre.