# Zajos CT-k
- 2
- 13
- 15
- 21
- 26
- 34
- 35
- 38
- 43
- 45
- 46

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