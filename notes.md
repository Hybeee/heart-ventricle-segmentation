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

# Other
- patient_0020 streaking artifact, majd detector-ral le kell checkolni (nem detektalja rendesen)
- Streaking artifact-okhoz erdekesseg: tobb helyen megjelennek, de csak patient_0045 eseten okoznak gondot. Mas esetben - pl. patient_0048 - nem. Ennek a miertje a polar plot-on jol latszik; szimplan elkeruli az nnunet a streaking artifact-ot, a sugar es a maszk boundary metszesenel nincsen extrem gradiens ugras.
- Multi-slice algoritmushoz megjegyzes: lehet erdemes lenne megnezni nem csak a maszkok minosege es a score-ok kozotti korrelaciot, hanem a threshold ertekek es a score-ok kozotti korrelaciot