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
- 