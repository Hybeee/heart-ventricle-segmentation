NOTE: A referenciamaszk az elozo, 1000 iteracioig futtatot GAC eredmenye

GAC_PARAMS = {
    "curvature_scaling": 1.00,
    "advection_scaling": 1.00,
    "propagation_scaling": 0.2976,
    "num_iterations": 1000,
    "max_rms_error": 0.0001
}

# patient_0001

A ket maszk kb. ugyanaz.

Result: OK

# patient_0002

Az uj maszk jobban rasimul az eredeti maszkra, nehol levagva reszeket, de ez szerintem nem rossz;

Az iteraciokat megvizsgalva konvergalt (a vegere mar minden d=50 iteracional csak par pixel valtozott adott szeleten, ami a szivizmot tartalmazta)

![p2_1](gac_results_images/patient_0002_1.png)

Result: OK

# patient_0003

A ket maszk kb. ugyanaz.

Result: OK

Update: 10k iteracional bubble jelenseg -> itt se jok a parameterek:(

# patient_0004

Az uj maszk kicsit jobban rasimul az eredetire. Itt viszont ez biztos, hogy jobb, mint a masik (ott a burok tulsagosan 'tulszegmental')

Result: OK

# patient_0005

A ket maszk kb. ugyanaz.

Result: OK

# patient_0006

Az eredeti maszk rossz, sok bemelyedes van benne, ami szerintem magyarazhatova teszi az eredmenyt. Itt is jobban 'alulszegmental' az uj maszk. Se az uj, se a regi nem jo, de ez inkabb az eredeti LV szegmentacio miatt van.

Erdekesseg: nem tunik ugy, mintha konvergalt volna. Itt eleinte a maszk alig valtozik, sot elkezd csokkenni, majd kesobb elkezd noni.

![p6_1](gac_results_images/patient_0006_1.png)

## Tobb iteraciora futtatva

Jonak tunik igy! Ugy tunik, hogy az utolso ket iteracio (5000 es 10000) alapjan konvergalt a maszk

![p6_2](gac_results_images/patient_0006_2.png)

# patient_0007

Az eredeti maszk nem tokeletes -> nincs benne jellegzetes bemelyedes. A maszk egyebkent konvergalt.

Az eredeti felveltel is erdekes, keves szelet, a koronalis szeleten nehezen meghatarozhato, hogy hol vannak a szivizmok (GT alapjan sem ertelmezheto)

![p7_1](gac_results_images/patient_0007_1.png)

# patient_0008

Uj maszk robban rasimul az LV maszkra, mint a regi, viszont emiatt a bemelyedeseket se tolti ki teljesen.

![p8_1](gac_results_images/patient_0008_1.png)

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

![p8_2](gac_results_images/patient_0008_2.png)

Tul nagy propagation scaling?
Tesztel scaling-ek (curv=adv=1.0):
- p0p0530 - 0
- p0p0706 - 1
- p0p0942 - 2
- p0p1255 - 3
- p0p1674 - 4
- p0p2232 - 5
- p0p2976 - 6

Kisebb prop eseten is kb. ugyanabba az iranyba konvergalnak -> pl. 2232 eseten kb latszik mar hogy ha tovabb tudna menni, akkor kialakulna a gomb -> simitani kell -> adv=1.0, prop=0.2976, curv:
- ...

# patient_0009

Itt is kozelebb van az eredeti maszkhoz az uj GAC maszk. Kerdeses, hogy a regi, vagy az uj jobb (pl. kep), a zold szivizmot az uj se veszi be teljesen -> jo-e itt a szivizom szegmentacioja?

Az uj maszk nem konvergalt.

TODO: TOBB ITERACIOIG FUTTATNI!!!!

![p9_1](gac_results_images/patient_0009_1.png)

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0010

Ugyanugy kozelebb van az eredeti maszkhoz. Itt viszont nem tudom pontosan megallapitani, hogy konvergalt-e mar vagy sem.

TODO: TOBB ITERACIO

![p10_1](gac_results_images/patient_0010_1.png)

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0011

Konvergal es jobban kimegy, mint a regi maszk. Erdekesseg, hogy a kepen levo szegmentacio axialis szeleten egy pupnak nez ki.

TODO: tobb iteracio

![p11_1](gac_results_images/patient_0011_1.png)
![p11_2](gac_results_images/patient_0011_2.png)

Egyebkent nem biztos, hogy konvergalt, ez is erdekes lehet. A szivizom formaja/bemelyedese itt is mas. GT nehezen ertelmezheto/indokolhato. Inkabb csak a helyezete alapjan azonosithato

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0012

Itt is elobb megall, viszont szerintem nem konvergalt.

TODO: TOBB ITERACIO

![p12_1](gac_results_images/patient_0012_1.png)

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0013

Rossz CT

# patient_0014

Alapvetoen jo, de megjelennek puklik.

TODO: Tobb iteracio, mert lehet, hogy nem konvergalt

![p14_1](gac_results_images/patient_0014_1.png)

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0015

A ket maszk kb. ugyanaz.

Result: OK

# patient_0016

Kicsit szerintem jobb, de egy pukli elkezdett megjelenni. Viszont konvergalt.

TODO: tobb iteracio, hogy BIZTOS konvergalt-e

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)


# patient_0017

Jo/jobb, mint a regi. Konvergalas kerdojeles

TODO: tobb iteracio

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)


# patient_0018

Konvergalt es jo

# patient_0019

Jobban rasimul az eredeti maszkra, ez egy helyen problemasnak tunik;

![p19_1](gac_results_images/patient_0019_1.png)

Konvergenciat erdemes ellenorizni.

TODO: tobb iteracio

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0020

Alapvetoen jobb, meg a streaking ellen robosztusabb is (nem szegmental tul), viszont emiatt az egyik izomnal beesik;

![p20_1](gac_results_images/patient_0020_1.png)

Szinte biztos, hogy nem konvergalt (viszont lassan indul!)

TODO: tobb iteracio

## Tobb iteraciora futtatva:

Rendesen konvergal!

![p20_2](gac_results_images/patient_0020_2.png)

# patient_0021

Szerintem nem konvergalt

TODO: tobb iteracio

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# patient_0022

Szukebben rajta van az eredeti maszkon, ami jo, viszont ez egy bizonyos szelet eseten rossz;

![p22_1](gac_results_images/patient_0022_1.png)

Iteraciokat megvizsgalva o se biztos, hogy konvergalt;

TODO: tobb iteracio

## Tobb iteraciora futtatva:

A ROI-hoz konvergal (vagy legalabbis 'kifolyik' a szivizom bemelyedesenel)

# Adaptive GAC
## patient_0001
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2300!
        Run took 89.4543s

Note: Good

## patient_0002
Output:

        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        Converged at iteration: 6350!
        Run took 288.1606s

Note: Good

## patient_0003
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2900
        Touches ROI wall: False
        Proposed rollback value: 900 - distance: 4.960150241851807
        Example voxel (z, y, x): (272, 204, 301)
        Performing rollback to 650 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        Converged at iteration: 3450!
        Run took 169.1358s

Note: Good

## patient_0004
Output:

        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        Converged at iteration: 3700!
        Run took 568.9609s

Note: Good

## patient_0005
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2600!
        Run took 188.0189s

Note: Good

## patient_0006
Output:

        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        At iteration: 7000
        Converged at iteration: 7050!
        Run took 654.1928s

Note: Rossz kezdeti LV maszk. Jo eredmeny.

## patient_0007
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2250!
        Run took 30.9468s

Note: Erdekes LV szegmentacio - legalabbis koronalis szeletet tekintve. Maszk alig mozdul -> Early convergence detection?

## patient_0008
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2450!
        Run took 213.8573s

Note: Konvergal egy olyan param set-tel, ami igazabol bubble-t eredmenyzene. "Good".

## patient_0009
Output:

        At iteration: 1000
        Bubble detected at iteration: 1800
        Touches ROI wall: True
        Proposed rollback value: 650 - distance: 4.747989654541016
        Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        Bubble detected at iteration: 3850
        Touches ROI wall: True
        Proposed rollback value: 850 - distance: 4.869833946228027
        Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 2.25
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        At iteration: 7000
        At iteration: 8000
        At iteration: 9000
        Converged at iteration: 9050!
        Run took 617.6615s

Note: Good, de egy helyen ugy latszik mintha kezdene kialakulni a lufi. Ez nem feltetlen(?) baj.

![p9_agac_res](gac_results_images/patient_0009_agac_res.png)

## patient_0010
Output:

        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        Bubble detected at iteration: 4150
        Touches ROI wall: True
        Proposed rollback value: 1000 - distance: 4.958131790161133
        Performing rollback to 750 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        At iteration: 7000
        Converged at iteration: 7700!
        Run took 897.0285s

Note: Good, de itt is kezd valami bubble forma latszani (patient_0009)
![p10_agac_res](gac_results_images/patient_0010_agac_res.png)

## patient_0011
Output:

        At iteration: 1000
        Bubble detected at iteration: 1600
        Touches ROI wall: True
        Proposed rollback value: 500 - distance: 4.997868061065674
        Performing rollback to 250 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2250!
        Run took 57.8648s

Note: Good

## patient_0012
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2750
        Touches ROI wall: True
        Proposed rollback value: 1000 - distance: 4.875160217285156
        Performing rollback to 750 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        Converged at iteration: 6500!
        Run took 280.8136s

Note: Good, de a bubble itt is latszik kb. Bar itt magyarazhato
![p12_agac_res](gac_results_images/patient_0012_agac_res.png)

## patient_0013
Output:

        At iteration: 1000
        Converged at iteration: 1550!
        Run took 42.4958s

Note: Rossz CT-n rossz LV maszk. De lefut - helyesen. Jo alap arra, hogy mi tortenik ezzel a paramset-tel, ha nincs bemelyedes (prop (+ curv?) eleg eros ahhoz, hogy a kezdeti helyen tartsa a maszkot).

## patient_0014
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2000
        Touches ROI wall: True
        Proposed rollback value: 450 - distance: 4.759858131408691
        Performing rollback to 200 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        Converged at iteration: 3300!
        Run took 160.5114s

Note: Goo

## patient_0015
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2550!
        Run took 122.7959s

Note: Good

## patient_0016
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2450
        Touches ROI wall: True
        Proposed rollback value: 800 - distance: 4.904951572418213
        Performing rollback to 550 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        Converged at iteration: 4550!
        Run took 156.4347s

Note: Good. Talan kicsi bubble?
![p16_agac_res](gac_results_images/patient_0016_agac_res.png)

## patient_0017
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2650
        Touches ROI wall: True
        Proposed rollback value: 650 - distance: 4.847691059112549
        Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2650!
        Run took 59.7538s

Note: Good (kicsit torzitott koronalis szelet, nehezebb megmondani)

## patient_0018
Output:

        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2850!
        Run took 92.7089s

Note: Good

## patient_0019
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2900
        Touches ROI wall: False
        Proposed rollback value: 800 - distance: 4.9019622802734375
        Example voxel (z, y, x): (333, 195, 412)
        Performing rollback to 550 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        Converged at iteration: 4400!
        Run took 253.0886s

Note: Good, kicsi bubble mintha kezdene kialakulni, talan magyarazhato?
![p19_agac_res](gac_results_images/patient_0019_agac_res.png)

## patient_0020
Output:

        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        At iteration: 7000
        Converged at iteration: 7500!
        Run took 1434.0654s

Note: Good. Tul lassu -> tul mely szivizmok miatt?

## patient_0021
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2550
        Touches ROI wall: False
        Proposed rollback value: 800 - distance: 4.766111850738525
        Example voxel (z, y, x): (418, 211, 341)
        Performing rollback to 550 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        Bubble detected at iteration: 4650
        Touches ROI wall: False
        Proposed rollback value: 1550 - distance: 4.865890979766846
        Example voxel (z, y, x): (419, 212, 341)
        Performing rollback to 550 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 2.25
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        Converged at iteration: 2850!
        Run took 291.0293s

Note: Tul hamar konvergal. Valoszinuleg 1.5 es 2.25 kozott van a jo prop value? Kell early convergence detection
![p21_agac_res](gac_results_images/patient_0021_agac_res.png)

## patient_0022
Output:

        At iteration: 1000
        At iteration: 2000
        Bubble detected at iteration: 2500
        Touches ROI wall: True
        Proposed rollback value: 450 - distance: 4.681895732879639
        Performing rollback to 200 of exp c1p0000_p0p2976_a1p0000
        New params:
                curvature_scaling: 1.50
                advection_scaling: 1.00
                propagation_scaling: 0.30
                num_iterations: 1000.00
                max_rms_error: 0.00
        At iteration: 1000
        At iteration: 2000
        At iteration: 3000
        At iteration: 4000
        At iteration: 5000
        At iteration: 6000
        Converged at iteration: 6750!
        Run took 289.7664s

Note: Good, de itt is a bubble dolog talan feljon? Bar itt szerintem foleg nem rossz.
![p22_agac_res](gac_results_images/patient_0022_agac_res.png)

## patient_0023
Output:
	At iteration: 1000
	Bubble detected at iteration: 1550
	Touches ROI wall: True
	Proposed rollback value: 900 - distance: 4.894726753234863
	Performing rollback to 650 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Converged at iteration: 4050!
	Run took 54.2924s
Note: Good, de
A masodik (AL) izmot nem talalja meg az algoritmus. Ennek az az oka, hogy a CT rossz -> nincs ott bemelyedes/minimalis -> GAC nem fedi be. A masik izom eseten jol fut
![p23_agac_res](gac_results_images/patient_0023_agac_res.png)
![p23_gt](gac_results_images/patient_0023_gt.png)

## patient_0024
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2650!
	Run took 79.5709s

Note: Good

## patient_0026
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3050
	Touches ROI wall: False
	Proposed rollback value: 950 - distance: 4.572650909423828
	Example voxel (z, y, x): (626, 230, 338)
	Performing rollback to 700 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Bubble detected at iteration: 5150
	Touches ROI wall: False
	Proposed rollback value: 1150 - distance: 4.827953338623047
	Example voxel (z, y, x): (627, 229, 339)
	Performing rollback to 700 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Converged at iteration: 5150!
	Run took 1351.6854s

Note: Good, szokasos kisebb pukli. Ez kb. kovetkezik a maszk alakjabol

## patient_0027
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3000
	Touches ROI wall: False
	Proposed rollback value: 900 - distance: 4.918724060058594
	Example voxel (z, y, x): (375, 208, 296)
	Performing rollback to 650 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	At iteration: 6000
	At iteration: 7000
	Converged at iteration: 7600!
	Run took 310.3336s

Note: Good

## patient_0028
Output:

	At iteration: 1000
	Bubble detected at iteration: 1400
	Touches ROI wall: True
	Proposed rollback value: 350 - distance: 4.945931911468506
	Performing rollback to 100 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3100
	Touches ROI wall: False
	Proposed rollback value: 500 - distance: 4.828218936920166
	Example voxel (z, y, x): (86, 199, 347)
	Performing rollback to 100 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3850!
	Run took 77.6429s

Note: Rossz CT -> a maszk alakja megint elfedi az egyik izmot (orvos maszknal is) -> GAC 'rosszul' fut / masik izmot egyebkent jol tomi be (AL).

## patient_0029
Output:

	At iteration: 1000
	Bubble detected at iteration: 1400
	Touches ROI wall: True
	Proposed rollback value: 350 - distance: 4.945931911468506
	Performing rollback to 100 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3100
	Touches ROI wall: False
	Proposed rollback value: 500 - distance: 4.828218936920166
	Example voxel (z, y, x): (86, 199, 347)
	Performing rollback to 100 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3850!
	Run took 78.0643s

Note: Ugyanaz a felvetel, mint 0028

## patient_0030
Output:

	At iteration: 1000
	Bubble detected at iteration: 1550
	Touches ROI wall: False
	Proposed rollback value: 300 - distance: 4.650102615356445
	Example voxel (z, y, x): (292, 213, 365)
	Performing rollback to 50 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2700
	Touches ROI wall: False
	Proposed rollback value: 350 - distance: 4.932283401489258
	Example voxel (z, y, x): (293, 205, 366)
	Performing rollback to 50 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Converged at iteration: 4650!
	Run took 511.0703s

Note: Good, itt talan kicsit tul nagy a konvergencia ellenere a buborek?

## patient_0031
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3200
	Touches ROI wall: False
	Proposed rollback value: 1000 - distance: 4.732285976409912
	Example voxel (z, y, x): (102, 200, 290)
	Performing rollback to 750 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Converged at iteration: 5700!
	Run took 131.0975s

Note: Szerintem hamarabb konvergalt, mint kellett volna. Lehet itt rosszabb az alap LV szegmentacio

## patient_0032
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2450!
	Run took 30.3718s

Note: Good - lenti, de a fentit nem lehet szegmentalni. A GT-nel sincs ott bemelyedes.

## patient_0033
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2450!
	Run took 30.4269s

Note: Ugyanaz, mint patient_0032

## patient_0034
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2150
	Touches ROI wall: False
	Proposed rollback value: 650 - distance: 4.681351184844971
	Example voxel (z, y, x): (280, 225, 363)
	Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Bubble detected at iteration: 4350
	Touches ROI wall: True
	Proposed rollback value: 850 - distance: 4.681351184844971
	Example voxel (z, y, x): (279, 223, 364)
	Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	At iteration: 6000
	At iteration: 7000
	At iteration: 8000
	Converged at iteration: 8900!
	Run took 363.5924s

Note: Good

## patient_0035
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3350!
	Run took 108.2891s

Note: Good

## patient_0036
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3700!
	Run took 51.9868s

Note: Good

## patient_0037
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2100!
	Run took 135.8744s

Note: Good

## patient_0038
Output:

	At iteration: 1000
	Converged at iteration: 1900!
	Run took 84.6016s

Note: Good

## patient_0039
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2700
	Touches ROI wall: True
	Proposed rollback value: 1000 - distance: 4.983532428741455
	Performing rollback to 750 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Converged at iteration: 5050!
	Run took 206.7093s

Note: Hasonlit az orvos maszkhoz, de ott nincs bemelyedes a PM-nel. AL-nel jo. Van egy pont, ahol bemelyed. Itt a baj az, hogy mar inicializacional 'osszemegy' a maszk. Ezt jo lenne megnezni reszletesen. Pl dilatalt verziobol inditva mi tortenik?

## patient_0040
Output:

	At iteration: 1000
	Bubble detected at iteration: 1400
	Touches ROI wall: False
	Proposed rollback value: 350 - distance: 4.870301246643066
	Example voxel (z, y, x): (73, 183, 344)
	Performing rollback to 100 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2600!
	Run took 63.3835s

Note: Good

## patient_0041
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2100
	Touches ROI wall: False
	Proposed rollback value: 300 - distance: 4.624691486358643
	Example voxel (z, y, x): (284, 215, 340)
	Performing rollback to 50 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Bubble detected at iteration: 5400
	Touches ROI wall: False
	Proposed rollback value: 350 - distance: 4.624691486358643
	Example voxel (z, y, x): (282, 214, 339)
	Performing rollback to 50 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	At iteration: 6000
	At iteration: 7000
	Converged at iteration: 7100!
	Run took 357.8926s

Note: Good

## patient_0042
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2100!
	Run took 24.9228s

Note: Egyik oldalon van csak lyuk. De jo

## patient_0043
Output:

	At iteration: 1000
	Converged at iteration: 1600!
	Run took 66.6199s

Note: Rossz CT

## patient_0044
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2100
	Touches ROI wall: False
	Proposed rollback value: 650 - distance: 4.938744068145752
	Example voxel (z, y, x): (335, 220, 373)
	Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Bubble detected at iteration: 4100
	Touches ROI wall: False
	Proposed rollback value: 900 - distance: 4.938744068145752
	Example voxel (z, y, x): (339, 209, 376)
	Performing rollback to 400 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	At iteration: 6000
	Converged at iteration: 6450!
	Run took 400.5884s

Note: Good

## patient_0045
Output:

	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2800!
	Run took 238.9081s

Note: Good

## patient_0046
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3800!
	Run took 172.0702s

Note: Good

## patient_0047
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3300
	Touches ROI wall: False
	Proposed rollback value: 450 - distance: 4.961279392242432
	Example voxel (z, y, x): (195, 254, 352)
	Performing rollback to 200 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	At iteration: 6000
	Bubble detected at iteration: 6750
	Touches ROI wall: False
	Proposed rollback value: 550 - distance: 4.961279392242432
	Example voxel (z, y, x): (194, 255, 353)
	Performing rollback to 200 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3450!
	Run took 367.8510s

Output: Good, talan kicsit mehetett volna tovabb, de jo. Slicer-ben nezve jo

## patient_0048
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Bubble detected at iteration: 3050
	Touches ROI wall: False
	Proposed rollback value: 900 - distance: 4.997561454772949
	Example voxel (z, y, x): (361, 236, 392)
	Performing rollback to 650 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	Converged at iteration: 2950!
	Run took 291.0459s

Note: Good

## patient_0049
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2500
	Touches ROI wall: True
	Proposed rollback value: 800 - distance: 4.7578125
	Performing rollback to 550 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Converged at iteration: 4650!
	Run took 78.1591s

Note: Good

## patient_0050
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2900
	Touches ROI wall: False
	Proposed rollback value: 850 - distance: 4.9534993171691895
	Example voxel (z, y, x): (214, 187, 402)
	Performing rollback to 600 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Bubble detected at iteration: 5450
	Touches ROI wall: False
	Proposed rollback value: 1000 - distance: 4.792359828948975
	Example voxel (z, y, x): (214, 187, 402)
	Performing rollback to 600 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Converged at iteration: 4600!
	Run took 458.3043s

Note: Good

## patient_0051
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2100
	Touches ROI wall: True
	Proposed rollback value: 1200 - distance: 4.80518102645874
	Performing rollback to 950 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	Converged at iteration: 3300!
	Run took 62.1986s

Note: Good

## patient_0052
Output:

	At iteration: 1000
	At iteration: 2000
	Bubble detected at iteration: 2350
	Touches ROI wall: True
	Proposed rollback value: 750 - distance: 4.9490766525268555
	Performing rollback to 500 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 1.50
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Bubble detected at iteration: 4600
	Touches ROI wall: True
	Proposed rollback value: 900 - distance: 4.829232215881348
	Performing rollback to 500 of exp c1p0000_p0p2976_a1p0000
	New params:
		curvature_scaling: 2.25
		advection_scaling: 1.00
		propagation_scaling: 0.30
		num_iterations: 1000.00
		max_rms_error: 0.00
	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	At iteration: 5000
	Converged at iteration: 5850!
	Run took 367.7309s

Note: Good

## patient_0053
Output:

	At iteration: 1000
	At iteration: 2000
	At iteration: 3000
	At iteration: 4000
	Converged at iteration: 4750!
	Run took 162.1879s

Note: Good

# kiertekeles
(LV, muscle -> legyen ilyen modell mindkettore egyszerre betanitva)
yuki vs nnunet
yuki vs threshold
szamok: dice, iou, Hausdorff distance
derivaltas kuszobolos cikk kell

residual vagy skip connection van-e az nnunet-ben (Gabornak uzenet)

# MUSCLE SEGMENTATION
## patient_0001
LV:
- jo, hasonlit az orvoshoz

Izom:
- AL: Jonak tunik. GT szerintem alulszegmental
- PM: Jonak tunik, kiveve a kepen (itt bizonytalan). Nehol tulszegmental az GT, valahol pedig nem (itt nem biztos, hogy kell, kep->). Itt az GT is picit beleszegmental. HU ertekeket nezve ez hatareset, hogy melyik kell/jo.

![p1_hms](gac_results_images/patient_0001_hms_1.png)

## patient_0002
LV:

- jo, hasonlit az GT-hez
- streaking utan a hid nagyon minimalis resze bennemarad. Szerintem ezt nem nagyon lehet kivedeni

Izom:

- AL: jo, az GT-vel kb. hasonlo. Egy ponton szerintem GT tulszegmental;
- PM: jo, hasonlo kb. GT-ben nem connected component ez az izom -> rossz(?)

![p2_hms](gac_results_images/patient_0002_hms_1.png)

## patient_0003
LV:
- Talan kicsit tulszegmental, mint GT. A CT-t elnezve ez nem feltetlen rossz
Izom:
- AL: Logika alapjan jo, GT teljesen rossz (lokacio)
- PM: Logika alapjan jo, GT teljesen rossz (lokacio)
- Note: Kesobbi megfigyeles alapjan - valszeg - a GT is jo, csak en rosszul irtam ki a maszkot.

![p3_hms](gac_results_images/patient_0003_hms_1.png)

## patient_0004
LV:
- Itt a streaking-et a postproc alg se tudta tokeletesen eltavolitani. Emiatt ott van elteres, egyebkent jo.

Izom:
- A streaking-es dolog miatt a konvex burok rossz -> lesz egy hatalmas komponens axialis szeleten a streaking-nel, ami emiatt 'izom' lesz. Egyebkent ez kiszurheto szerintem. A GT AL esetben alul, PM esetben tulszegmental szerintem.

![p4_hms](gac_results_images/patient_0004_hms_1.png)

## patient_0005
LV:
- jo, kb. ugyanazok
- streaking egy ponton 'belevag' a maszkba. Ezt egyebkent a konvex burok befedi

Izom:
- AL: jo, GT alulszegmental
- PM: jo, GT alulszegmental
- Megjegyzes: Mostani ellenorzesben diff-et morph. close-oltam. Ugy az AL izom szetvalt ket kulon komponensre -> ezek lettek a legnagyobbak. A morph closing elhagyasaval megoldodott a problema, viszont igy lesznek reszek PM eseten, akik 'gyengen' kapcsolodnak a fo strukturahoz -> adott DT erteku hidak torlese -> legnagyobb komponensek megtartasa? CT-t elnezve nem baj.

![p5_hms](gac_results_images/patient_0005_hms_1.png)

## patient_0006
LV:
- Erdekesebb CT, a postproc a kepen lathato okbol levag neha darabokat. Viszont jobb a szegmentacio, mint a GT (befalazza az izmokat)

![p6_hms](gac_results_images/patient_0006_hms_1.png)

Izmok:
- AL: Lokaciora jo, GT alulszegmental, de jobban. Tul nagy az izom merete de ez jon az LV szegmentaciobol
- PM: Lokaciora jo, meretre is. Hasonlit GT-re.

## patient_0007
LV:
- jo, hasonlit GT-re

Izmok:
- AL: -
- PM: -
- Note: Az LV maszk jo, viszont nincs/alig van bemelyedese. Egyedul AL-nel lathato bemelyedes, viszont a GAC az inicializalt helyzeteben maradt. Kene early conv detection -> nagyobb prop value, ez lehet megoldja. Ugyanakkor csak AL-t, PM-nek nincs (GT-ben) sem bemelyedes.

![p7_hms](gac_results_images/patient_0007_hms_1.png)

## patient_0008
LV:
- jo, ugyanaz, mint GT

Izom:
- AL: jo, GT szerintem tulszegmental
- PM: jo, GT szerintem tulszegmental

![p8_hms](gac_results_images/patient_0008_hms_1.png)

## patient_0009
LV:
- jo, GT-vel ugyanaz

Izom:
- AL: jo, viszont van egy kerdojeles resz (kep). Ezek 3D-ben connected-ek.
- PM: jo lokaciora
- Note: van egy problema. A ket betomott resz egy ponton osszeer. (kepen kijelolve). Emiatt azokat egy komponensek veszi - es egy masik kisebb komponens bejon. A maszk tulajdonsaga miatt szerintem a konvex burok tulsagosan kijon es ez biztositja par szeleten az atnyulast. Valahogy szet kene vagni. Otlet: Izmok lokaciojanak kijelolese: koronalis szeleten kozeppont felett/alatt, feluton BIZTOS elvagva. vagy valami ilyesmi. Itt egyeblent elobb megallitott GAC-nal is connected-ek. Max akkor nem, amikor a konvex burok 'rossz'.

![p9_hms](gac_results_images/patient_0009_hms_1.png)

## patient_0010
LV:
- jo, GT viszont befalazza az izmokat

Izom:
- AL: jo, GT alulszegmental
- PM: jo, GT alulszegmental

## patient_0011
LV:
- jo, GT jobban befalazza a szivizmot

Izom:
- AL: jo, a GT rosszabbnak tunik
- PM: okes, lehet rossz? GT nem nagyon ertelmezheto. Logikailag jo
- Note: itt se volt jo a diff morph close-olasa -> anelkul futtatva a kepek lent. PM-tol jobbra is diff resze (a kepen), de nem connected -> nem kerul be.

![p11_hms](gac_results_images/patient_0011_hms_1.png)
![p11_hms](gac_results_images/patient_0011_hms_2.png)

## patient_0012
LV:
- jo, hasonlitanak GT-vel

Izom:
- AL: jo
- PM: jo
- Note: Szerintem GT pontatlanabb, de hasonlitanak

## patient_0013
- Note: Rossz CT. Diff = ures -> nincs szivizom.
![p13_hms](gac_results_images/patient_0013_hms_1.png)

## patient_0014
LV:
- jo, talan a befalazasnal igaza van a GT-nek

Izom:
- AL: Jo, GT fura (kep), de lokaciora ugyanott
- PM: Jo, GT fura (kep), de lokaciora ugyanott

![p14_hms](gac_results_images/patient_0014_hms_1.png)
![p14_hms](gac_results_images/patient_0014_hms_2.png)

## patient_0015
LV:
- jo, kb ugyanaz, algo nagyobb maszkot dob vissza, de nem rosszabb/jobb egyertelmuen

Izom:
- AL: GT-hez kepest alulszegmental, de ez kovetkezik az LV szegmentaicobol
- PM: egy 3. komponens(kep, rossz lokacio) lett az 1./2. legnagyobb -> emiatt rossz. De a helyes maszk benne van diff-ben.

![p15_hms](gac_results_images/patient_0015_hms_1.png)

## patient_0016
LV:
- jo, jobb, mint GT a szivizmoknal (befalazas kb)

Izom:
- AL: Jo
- PM: Jo
- Note: GT ugyanugy rossz helyen van, viszont ez szerintem a kodom miatt van (rossz orientacio? le kell majd checkolni). Helyes szegmentacio a kepen, amibol latszik, hogy GT alulszegmental
![p16_hms](gac_results_images/patient_0016_hms_1.png)

## patient_0017
LV:
- Jo, kb. ugyanaz, mint GT

Izom:
- AL: jo
- PM: jo, de a GT alulszegmental
- Note: Morph closing nem kell

## patient_0018
LV:
- jo, kb ugyanaz, mint GT

Izom:
- AL: jo
- PM: jo, GT tulszegmental
- Note: Itt is latszik, hogy morph close nem kell/mas kell helyette
![p18_hms](gac_results_images/patient_0018_hms_1.png)

## patient_0019
LV:
- jo, GT tulszegmental nagyon

Izom:
- AL: jo
- PM: jo
- Note: GT LV tulszegmentalasa itt kihat -> foleg PM erdekes, de AL is.
![p19_hms](gac_results_images/patient_0019_hms_1.png)

## patient_0020
LV:
- jo, hasonlo, mint GT. Mini kinyulasok, mint patient_0006-nal -> emiatt a kepen levo szegmentacio elofordul

![p20_hms](gac_results_images/patient_0020_hms_1.png)

Izom:
- AL: jo
- PM: jo
- Note: Az alabb lathato kepen GT beveszi a belso lyukat az izomba, az algoritmus nem. A belso resz diff resze, de valoszinuleg a kimeneti PM izommal nincs osszekotve
![p20_hms](gac_results_images/patient_0020_hms_2.png)

## patient_0021
LV:
- jo, kb. ugyanaz, mint GT

Izom:
- AL: lokacio jo
- PM: lokacio jo
- Note: GAC itt tul hamar konvergal -> alulszegmentalt izmok
![p21_hms](gac_results_images/patient_0021_hms_1.png)

## patient_0022
LV:
- Alapvetoen jo, de postproc alg levag kisebb szarakat;
![p22_hms](gac_results_images/patient_0022_hms_1.png)

Izom:
- AL: jo
- PM: jo
- Note: GT nagyon rossz
![p22_hms](gac_results_images/patient_0022_hms_2.png)

## patient_0023
LV:
- jo, GT nagyon tulszegmental

![p23_hms](gac_results_images/patient_0023_hms_1.png)

Izom:
- AL: -
- PM: jo
- Note: Nincs eleg curve fent -> arra nem no a GAC maszk -> nincs izom ott. Kellene valami fallback?

![p23_hms](gac_results_images/patient_0023_hms_2.png)

## patient_0024
LV:
- jo, hasonlo GT-hez. GT alulszegmental algo-hoz kepest

Izom:
- AL: jo, bar GT tobbet szegmental (alulszegmentalas miatt?)
- PM: morph close-t hasznalva rossz helyen (elso kep)
- Note: morph close nelkul jo a szegmentacio (masodik kep)

![p24_hms](gac_results_images/patient_0024_hms_1.png)
![p24_hms](gac_results_images/patient_0024_hms_2.png)

## patient_0026
LV:
- jo, kb. ugyanaz, mint GT

Izmok:
- AL: jo
- PM: jo
- Note: AL-hez van egy kapcsolod resz, ami a GT-ben nincs benne. Jo kerdes, hogy ez szivizom-e. Tippre nem, mert tulsagosan 'kozeppen' van az axialis szeleteket tekintve. Kene valahogy szurni. Emellett GT erdekes, foleg a PM. Szerintem rossz.

![p26_hms](gac_results_images/patient_0026_hms_1.png)

## patient_0027
LV:
- jo, hasonlo, mint a GT, de itt is megemlitheto a 'kinyulas'

Izmok:
- AL: jo
- PM: jo
- Note: Szerintem orvos tulszegmental AL-t, PM-t meg alul

![p27_hms](gac_results_images/patient_0027_hms_1.png)

## patient_0028
LV:
- jo, kb. olyan, mint a GT

Izmok:
- AL: jo
- PM: kerdeses
- Note: Itt meg jobban latszik, hogy miert nem kell morph close. Illetve PM lokacioja erdekes. Lehet a GT a jo, de mintha a ketto egyszerre lenne jo? (3D-ben nem.) Ha valasztani kell, akkor szerintem GT lokacio jobb. Ahol a GT van ott viszont nincs bemelyedes.

## patient_0029
Note: Ugyanaz a felvetel, mint patient_0028

## patient_0030
LV:
- jo, szinte ugyanaz, mint GT

Izmok:
- AL: jo
- PM: jo
- Note: Talan itt a buborek hatasa jobban latszik sajnos.

![p30_hms](gac_results_images/patient_0030_hms_1.png)

## patient_0031
LV:
- algo tulszegmental GT-hez kepest. Axialis szeleteket nezve nem feltetlen rossz

Izmok:
- AL: jo
- PM: jo
- Note: GT maszk csakis GT LV-hez kepest jo, illetve itt is csak morph closing nelkul volt jo az eredmeny

## patient_0032
LV:
- jo, hasonlo, mint GT

Izmok:
- Note: Itt is csak egyik oldalt no meg a GAC altal visszaadott maszk. Emiatt nincs AL izom. PM egyebkent jo. Talan kicsit pup? De kornyezo szeleteket megnezve magyarazhato

![p32_hms](gac_results_images/patient_0032_hms_1.png)

## patient_0033
Note: ugyanaz a felvetel, mint patient_0032

## patient_0034
LV:
- zajos felvetel, hasonlo, mint GT (~patient_0002)

Izmok:
- AL: jo
- PM: alulszegmental? GT tul, biztosan

![p34_hms](gac_results_images/patient_0034_hms_1.png)

## patient_0035
LV:
- zajosabb CT, hasonlo, mint GT

Izmok:
- AL: jo(?), GT-hez kepest alulszegmental. CT alapjan nehez megmondani, hogy melyik a jo
- PM: Rossz lokacion van az 1./2. largest CC -> rossz szivizom -> lokacio alapu szures kell

![p35_hms](gac_results_images/patient_0035_hms_1.png)

## patient_0036
LV:
- jo, GT mintha alul szegmentalna, de nehez eldonteni, hogy melyik jobb

Izmok:
- AL: jo
- PM: jo
- Note: Egyreszt itt is latszik, hogy nem kell closing. Masreszt a lokacio GT-hez kepest jo, viszont GT az alap LV maszk miatt lehet rosszabb?

## patient_0037
LV:
- jo, hasonlo, mint GT

Izmok:
- AL: jo
- PM: jo
- Note: Kepen lathato resznel van elteres. A diff-ek nem connected component-ek, ezert nincs az AL maszkban. Ez valoszinuleg a morph closing miatt van. Update: amiatt van. Anelkul a maszkok szinte ugyanazok

![p37_hms](gac_results_images/patient_0037_hms_1.png)

## patient_0038
LV:
- rosszabb CT, maszk kb. jo. Jobban 'befalaz', mint GT, de van lyuk az AL izomnak (axialis)

Izmok:
- AL: jo
- PM: jo
- Note: morph close itt is beleront

## patient_0039
LV:
- hasonloak, kiveve az AL izom befalazasanal. Ott nem rossz a GT, a postproc algoritmus szedi le a 'fal' egy reszet. Ez par adott szeleten kicsi hiba

Izmok:
- AL: jo
- PM: Nincs bemelyedes/minimalis -> GAC nem tolti ki.

![p39_hms](gac_results_images/patient_0039_hms_1.png)

## patient_0040
LV:
- Hasonlo, mint GT

Izmok:
- AL: jo
- PM: jo
- Note: morph closing megint nem kell

## patient_0041
LV:
- hasonlo, mint GT

Izmok:
- AL: jo
- PM: lehet kicsit alulszegmental? 3D-ben nincsenek osszekotve
- Note: Erdekes resz: slicer 61.9231mm kornyeken levo szeletek

## patient_0042
LV:
- Bennemarad leak. Ezenkivul kb 1 voxellel tulszegmental a GT, mindketto jonak tunik

Izmok:
- AL: pozicio jo
- PM: pozicio jo
- Note: Egyreszt ez csak morph closing nelkul. Masreszt AL eseten a maszk vegigfolyik az oldalan az LV maszknak -> ossze vannak kotve, emiatt is kene valami plusz itt meg

## patient_0043
- Note: rossz CT, mint patient_0013

## patient_0044
LV:
- GT maszk itt valamiert az AL izom (valoszinuleg rosszul olvastam ki). Jonak nez ki

Izmok:
- AL: tul sokaig megy le, de jo
- PM: jo

![p44_hms](gac_results_images/patient_0044_hms_1.png)

## patient_0045
LV:
- kicsit belevag a streaking sajnos, de jo, hasonlo, mint GT

Izmok:
- AL: jo
- PM: jo

## patient_0046
Note: Rossz CT -> rossz kimenet (~patient_0013)

## patient_0047
LV:
- jo, hasonlo GT-hez

Izmok:
- AL: jo
- PM: jo

## patient_0048
LV:
- jo

Izmok:
- AL: jo
- PM: jo
- Note: kis elteresek vannak, illetve lehet ez is a morph closing miatt. Nem, amugy sincsenek 3D-ben osszekotve

## patient_0049
LV:
- jo

Izmok:
- AL: jo
- PM: jo
- Note: GT tulszegmental algohoz kepest, bar nem tudom biztosra, hogy melyik a jo

## patient_0050
LV:
- GT-hez kepest kicsit alulszegmental, de CT-t nezve ez nem hiba. Mas axialis szeleteken a GT rossznak tunik.

Izmok:
- AL: jo
- PM: jo
- Note: Lokaciora es meretre jok. GT rossz helyen - szerintem itt is orientacios/kiolvasos dolog van a hatterben

## patient_0051
LV:
- hasonloak, viszont AL izom egyiknel se latszik/befalazzak. Rosszabb minosegu CT egyebkent

Izmok:
- AL: lokaciora jo, viszont lehet, mivel 'vastagabb' az algo fala -> nem er ki addig, mint GT. GT se tokeletes, illetve szerintem morph closing itt beleszol
- PM: jo
- Note: Closing-ot kiveve jobb az AL maszk, viszont tulsagosan 'kinyul' balra

## patient_0052:
LV:
- jo

Izmok:
- AL: talan kicsit tulsagosan sokaig er, de 3D-ben ossze van kotve a tobbi resszel. Egyebkent hasonlo, mint GT
- PM: jo

## patient_0053
LV:
- jo

Izmok:
- AL: -
- PM: Jo
- Note: PM oldalan ket nagy komponens is van. Sajnos ezeket valogatja be az algoritmus. AL izom is diff resze, jo az alakja, viszont nem eleg nagy. Morph closing elhagyasa megoldja.

![p53_hms](gac_results_images/patient_0053_hms_1.png)