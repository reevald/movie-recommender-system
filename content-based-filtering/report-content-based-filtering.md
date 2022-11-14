# Laporan Proyek Machine Learning - Moch Galang Rivaldo

<div style="padding:5px; background-color: #645CAA; text-align: center; font-size: 16pt; border-radius:10px;">Table of Contents</div>
<br/>

- [Project Overview](#project-overview)
- [Business Understanding](#business-understanding)
  - [Problem Statements](#problem-statements)
  - [Goals](#goals)
  - [Solution statements](#solution-statements)
- [Data Understanding](#data-understanding)
  - [Deskripsi Variabel & Attribute](#deskripsi-variabel--attribute)
  - [Menangani Missing Value](#menangani-missing-value)
  - [Menangani Outliers](#menangani-outliers)
  - [Univariate Analysis](#univariate-analysis)
    - [Fitur movieId](#fitur-movieid)
    - [Fitur title](#fitur-title)
    - [Fitur genres](#fitur-genres)
- [Data Preparation](#data-preparation)
  - [One-Hot Encoding](#one-hot-encoding)
- [Modeling](#modeling)
  - [Jaccard Similarity (Ground Truth)](#jaccard-similarity-ground-truth)
  - [Cosine Similarity](#cosine-similarity)
  - [Euclidean Similarity](#euclidean-similarity)
- [Evaluation](#evaluation)
  - [Sampel Film untuk Evaluasi](#sampel-film-untuk-evaluasi)
  - [Top-k Rekomedasi](#top-k-rekomedasi)
    - [Dengan Jaccard Similarity (Ground Truth)](#dengan-jaccard-similarity-ground-truth)
    - [Dengan Cosine Similarity](#dengan-cosine-similarity)
    - [Dengan Euclidean Similarity](#dengan-euclidean-similarity)
  - [Skor Metrik (nDCG@k)](#skor-metrik-ndcgk)
  - [Komparasi Waktu Komputasi](#komparasi-waktu-komputasi)
- [Conclusion](#conclusion)
- [Daftar Referensi](#daftar-referensi)

## Project Overview

Perusahaan penyedia layanan *streaming* film membutuhkan sistem rekomendasi yang dapat diterapkan ke dalam platform mereka. Kondisi platform tersebut masih baru *launching* dan masih sedikit user yang menggunakan. Tim R&D perusahaan tersebut berhasil mengatasi permasalahan *cold start* dengan membuat sistem rekomendasi *content-based filtering*. Ide dibalik sistem rekomendasi tersebut adalah dengan merekomendasikan film berdasarkan genre. Kemudian untuk menentukan urutan hasil rekomendasi, tim R&D menggunakan metode *Jaccard Similarity* untuk mengukur skor relevansi antar film. Metode tersebut dinilai cocok untuk mengevaluasi persamaan dan perbedaan genre antar film, terutama jika film memiliki banyak genre. Di sisi lain, tim R&D menemukan kekurangan metode *Jaccard Similarity* dalam hal menangani data yang berukuran besar. Jika data besar dikenakan metode tersebut maka waktu eksekusinya akan cukup lama dan memakan banyak *memory* (RAM). Hal ini membuat metode tersebut kurang menunjang pertumbuhan bisnis yang berkelanjutan.

Penelitian yang dilakukan Dooms, dkk. (2014) dengan judul "*In-memory, distributed content-based recommender system*" melakukan optimasi kecepatan eksekusi dan penggunaan *memory* (RAM) pada metode *Jaccard Similarity* dengan sistem kerja *parallel* dan *memory caching* [[1] (Hal 649-650)](https://link.springer.com/article/10.1007/s10844-013-0276-1#citeas). Solusi yang ditawarkan pada penelitian tersebut membutuhkan biaya pembangunan infrastruktur *hardware* yang tidak sedikit. Oleh sebab itu, tim R&D memilih untuk meneliti metode lain yang diharapkan dapat lebih cepat dan mempunyai hasil rekomendasi yang mendekati metode *Jaccard Similarity*. Pada proyek ini, hasil rekomendasi dengan metode Jaccard Similarity akan digunakan sebagai *ground truth* untuk perbandingan dalam mengevaluasi performa metode lain.

## Business Understanding

### Problem Statements

* Perusahaan penyedia layanan *streaming* film membutuhkan model terbaik dengan metode yang lebih cepat dan mampu menghasilkan hasil rekomendasi yang mendekati hasil rekomendasi dengan *Jaccard Similarity (ground truth)*.

### Goals

* Membangun model terbaik dengan metode yang lebih cepat dan mampu menghasilkan hasil rekomendasi yang mendekati hasil rekomendasi dengan *Jaccard Similarity (ground truth)*. Sehingga diharapkan dapat menunjang pertumbuhan bisnis sekaligus meminimalisir pengularan biaya  dalam pengembangan sistem rekomendasi.

### Solution statements
* Menawarkan solusi sistem rekomendasi dengan jenis *content-based filtering* yang menghasilkan rekomendasi film berdasarkan genre yang relevan. Untuk mendapatkan solusi terbaik, akan digunakan dua model yang berbeda yaitu *Cosine Similarity* dan *Euclidean Similarity*. Selain itu, untuk mengukur kinerja model akan digunakan metrik *Normalized Discounted Cumulative Gain* $(\text{nDCG@}k)$ pada top-k hasil rekomendasi terhadap *ground truth*.

## Data Understanding
**Sumber dataset**: [MovieLens (ml-latest-small.zip) - grouplens.org](https://grouplens.org/datasets/movielens/latest/)  
**Info dataset**: [README.html (ml-latest-small.zip) - grouplens.org](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)  
**Abstrak**: Dataset MovieLens pertama kali rilis pada tahun 1998 yang mendeskripsikan preferensi orang lain terhadap film [[2] (Hal 1)](http://files.grouplens.org/papers/harper-tiis2015.pdf). Versi dataset terbaru (*MovieLens Latest Datasets*) terdiri dari 9742 judul *movie* atau film yang berbeda yang di *generate* pada tanggal 26 September 2018. Dataset terdiri dari `links.csv`, `movies.csv`, `ratings.csv`, dan `tags.csv`. Pada kasus ini, dataset yang akan digunakan pada sistem rekomendasi *content-based filtering* hanya berfokus pada berkas `movies.csv`.

Tabel 1. Informasi Dataset pada `movies.csv`
| | *Description* |
|---|---|
| *Dataset Characteristics* | *Univariate* |
| *Attribute Characteristics* | *Integer, Categorical* |
| *Associated Tasks* | *Content-based filtering* |
| *Number of Instances* | 9742 |
| *Number of Attributes* | 3 |

### Deskripsi Variabel & Attribute
Berdasarkan informasi dari sumber dataset, variabel & attribute pada dataset *MovieLens Latest Datasets* (`movies.csv`) adalah sebagai berikut:
*  `movieId` (tipe: *numerical*) adalah identitas (Id) *movie* atau film yang berupa bilangan bulat yang berbeda dengan yang lainnya.
* `title` (tipe: *categorical*) adalah judul film yang diimpor dari https://www.themoviedb.org/.
* `genres` (tipe: *categorical*) adalah sebuah klasifikasi atau jenis dari film.

Berikut contoh lima data teratas pada dataset:

Tabel 2. Tampilan Lima Teratas pada Dataset
|index|movieId|title|genres|
|---|---|---|---|
|0|1|Toy Story \(1995\)|Adventure&#124;Animation&#124;Children&#124;Comedy&#124;Fantasy|
|1|2|Jumanji \(1995\)|Adventure&#124;Children&#124;Fantasy|
|2|3|Grumpier Old Men \(1995\)|Comedy&#124;Romance|
|3|4|Waiting to Exhale \(1995\)|Comedy&#124;Drama&#124;Romance|
|4|5|Father of the Bride Part II \(1995\)|Comedy|

### Menangani Missing Value
Untuk mendeteksi *missing value* digunakan fungsi `isnull().sum()` dan diperoleh sebagai berikut.

Tabel 3. Hasil Deteksi *Missing Value*
| Fitur | Jumlah *Missing Value* |
|:---:|:---:|
| `movieId` | 0 |
| `title` | 0 |
| `genres` | 0 |

Dari Tabel 3. terlihat bahwa setiap fitur tidak memiliki *Missing Value* (`NULL`) sehingga dapat dilanjutkan ke tahapan selanjutnya yaitu menangani *outliers*.

### Menangani Outliers
Secara intuitif salah satu ciri titik *outlier* yang `ideal` dalam data kategorikal adalah nilainya sangat jarang muncul. Kelangkaan nilai tersebut dapat diukur dengan menghitung frekuensi kemunculan setiap data kategorikal dalam dataset [[3] (Hal 3299)](https://www.eecs.ucf.edu/georgiopoulos/sites/default/files/247.pdf). Untuk implementasinya akan dilakukan pengecekan frekuensi kemunculan setiap genre (data kategorikal).

Tabel 4. Daftar Lima Genre dengan Frekuensi Kemunculan Terendah
| Genre | Frekuensi |
|---|:---:|
| (no genres listed) | 34 |
| Film-Noir | 87 |
| IMAX | 158 |
| Western | 167 |
| Musical | 334 |

Berdasarkan Tabel 4. diperoleh informasi bahwa genre `(non genres listed)` adalah yang paling jarang muncul. Dengan total 34 judul film dengan genre `(no genres listed)`.

Jika mengacu pada definisi outlier `ideal` (pernyataan sebelumnya) maka data 34 film dengan genre `(no genres listed)` dapat dinyatakan sebagai *outlier*. Namun *outlier* tersebut tidak akan dihapus karena sistem rekomendasi yang akan dibuat akan mendukung genre `(non genres listed)`. Jika dihapus maka 34 film tersebut menjadi kurang mendapatkan sorotan, karena tidak akan pernah muncul dalam daftar rekomendasi. Tentunya hal ini akan menurunkan nilai jual produk film.

### Univariate Analysis
Selanjutnya, akan dilakukan proses analisis pada setiap fitur pada dataset. 

#### Fitur movieId
Pada fitur `movieId` seharusnya tiap film memiliki identitas yang berbeda dengan yang lainnya, oleh karena itu pada tahap ini akan dicek duplikasi data pada `movieId`.

Tabel 5. Perbandingan Banyak *Unique* `movieId` dengan Total Data
|index|Unique movieId|Total Data|
|---|---|---|
|Jumlah Film|9742|9742|

Berdasarkan Tabel 5. didapat bahwa jumlah `movieId` yang berbeda (*unique*) sama dengan total data. Hal ini menunjukkan bahwa pada fitur `movieId` tiap film memiliki identitas (Id) yang berbeda-beda.

#### Fitur title
*Title* atau judul pada film seharusnya berbeda-beda (tidak identik satu sama lain) namun berdasarkan [info dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html#:~:text=Errors%20and%20inconsistencies), dimungkinkan terjadi kesalahan dan inkonsistensi pada judul film.

Tabel 6. Perbandingan Banyak *Unique* `title` dengan Total Data
|index|Unique title|Total Data|
|---|---|---|
|Jumlah Film|9737|9742|

Berdasarkan Tabel 6. jumlah `title` yang berbeda (*unique*) tidak sama dengan total data dengan selisih 9742 - 9737 = 5 *data point*. Untuk menindaklanjuti temuan tersebut, perlu dilakukan pengecekan data dengan `title` yang duplikat.

Tabel 7. Data Film dengan `title` yang Duplikat
|index|movieId|title|genres|
|---|---|---|---|
|650|838|Emma \(1996\)|Comedy&#124;Drama&#124;Romance|
|2141|2851|Saturn 3 \(1980\)|Adventure&#124;Sci-Fi&#124;Thriller|
|4169|6003|Confessions of a Dangerous Mind \(2002\)|Comedy&#124;Crime&#124;Drama&#124;Thriller|
|5601|26958|Emma \(1996\)|Romance|
|5854|32600|Eros \(2004\)|Drama|
|5931|34048|War of the Worlds \(2005\)|Action&#124;Adventure&#124;Sci-Fi&#124;Thriller|
|6932|64997|War of the Worlds \(2005\)|Action&#124;Sci-Fi|
|9106|144606|Confessions of a Dangerous Mind \(2002\)|Comedy&#124;Crime&#124;Drama&#124;Romance&#124;Thriller|
|9135|147002|Eros \(2004\)|Drama&#124;Romance|
|9468|168358|Saturn 3 \(1980\)|Sci-Fi&#124;Thriller|

Dari Tabel 7. diperoleh bahwa terdapat data yang duplikat. Untuk menanganinya dengan menghapus data `title` yang duplikat dan menyisakan `title` dengan `genres` yang lebih banyak (variatif). Karena dengan menyisakan film dengan genre yang lebih beragam akan meningkatkan relevansi dengan lebih banyak film lain sesuai dengan kemiripan genrenya.

Berikut perubahan banyak data setelah penghapusan data dengan `title` yang duplikat.

Tabel 8. Perbandingan Banyak *Unique* `title` dengan Total Data Setelah Penghapusan Data Duplikat
|index|Unique title|Total Data|
|---|---|---|
|Jumlah Film|9737|9737|

Berdasarkan Tabel 8. sekarang sudah tidak ada data dengan `title` yang duplikat. Banyak total data berubah dari 9742 menjadi 9737.

#### Fitur genres
Sebelumnya pada tahap menangani *outlier*, telah disebutkan daftar lima genre dengan frekuensi kemunculan terendah. Pada tahap ini akan melakukan eksplorasi distribusi frekuensi kemunculan setiap genre. Untuk mempermudah dalam menganalisa, akan digunakan fungsi `barplot()` dengan *output* sebagai berikut.

![](https://aneechan.github.io/assets/picture/mlt-s2/frekuensi-kemunculan-genre.png)

Gambar 1. Frekuensi Kemunculan Genre Film

Dari hasil visualisasi Gambar 1. di atas, genre yang sering muncul pada film adalah `Drama` diikuti `Comedy` dan `Thriller`. Sedangkan genre yang jarang muncul pada film adalah `(non genres listed)` dan cukup jelas bahwa pada dataset MovieLens ada 20 jenis genre sebagai berikut:
* Action
* Adventure
* Animation
* Children's
* Comedy
* Crime
* Documentary
* Drama
* Fantasy
* Film-Noir
* Horror
* IMAX
* Musical
* Mystery
* Romance
* Sci-Fi
* Thriller
* War
* Western
* (no genres listed)

Hal ini sedikit berbeda dengan jenis genre yang tertera pada [sumber dataset (grouplens.org)](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html#:~:text=Errors%20and%20inconsistencies) yang tidak menuliskan genre `IMAX` (jenis film dengan *high-resolution cameras*).

Selanjutnya akan dilakukan eksplorasi distribusi banyak genre setiap film menggunakan `barplot()` dengan *output* sebagai berikut.

![](https://aneechan.github.io/assets/picture/mlt-s2/banyak-genre-tiap-film.png)

Gambar 2. Banyak Genre Tiap Film

Dari hasil visualisasi Gambar 2. di atas, mayoritas film memiliki 1 hingga 3 genre yang berbeda. Sedangkan untuk banyak genre maksimal pada satu film di dataset adalah 10 genre yang berbeda. 

## Data Preparation
Pada tahap ini, akan dilakukan persiapan data ke dalam bentuk matrik korelasi antara judul film dengan genre sebelum diproses oleh model. Nilai setiap elemen pada matrik tersebut berupa angka 1 (ada) atau 0 (tidak ada). Contoh pada film A mempunyai genre Drama dan tidak memiliki genre Komedi maka untuk elemen (A, Drama) bernilai 1 sedangkan elemen (A, Komedi) bernilai 0.

Dengan kata lain, pada tahap ini akan dilakukan metode *`One-Hot Encoding`*` (OHE)` dari fitur kategorikal (genre) menjadi numerik. Metode ini dipilih karena pada fitur genre sudah memiliki batasan nilai (20 genre yang sudah ditentukan) dan setiap genrenya memiliki bobot yang sama walaupun berbeda frekuensi kemunculannya. Kesamaan bobot setiap genre dapat diwakilkan dengan data *boolean* (1 atau 0).

Hal itulah yang membuat metode seperti `Count Vectorizer` dan `TF-IDF Vectorizer` kurang sesuai untuk kasus ini.

### One-Hot Encoding
Berikut contoh lima data teratas hasil dari proses *One-Hot Encoding*.

Tabel 9. Contoh Lima Data Teratas dari Proses *One-Hot Endoding*
|title|\(no genres listed\)|Action|Adventure|Animation|Children|...|Romance|Sci-Fi|Thriller|War|Western|
|---|---|---|---|---|---|---|---|---|---|---|---|
|Toy Story \(1995\)|0|0|1|1|1|...|0|0|0|0|0|
|Jumanji \(1995\)|0|0|1|0|1|...|0|0|0|0|0|
|Grumpier Old Men \(1995\)|0|0|0|0|0|...|1|0|0|0|0|
|Waiting to Exhale \(1995\)|0|0|0|0|0|...|1|0|0|0|0|
|Father of the Bride Part II \(1995\)|0|0|0|0|0|...|0|0|0|0|0|

Dari hasil *output* di atas menunjukkan film Toy Story (1995) memiliki genre (*Adventure, Animation, Children*). Hal ini terlihat dari nilai elemen matriks 1 pada genre *Adventure, Animation, Children*. Selanjutnya, film Jumanji (1995) memiliki genre (*Adventure, Children*). Demikian seterusnya.

## Modeling
Pada tahap ini, untuk sistem rekomendasi yang dibuat akan menggunakan model dengan metode *Cosine Similarity* dan *Euclidean Similarity*. Sedangkan metode *Jaccard Similarity* dipilih sebagai metode untuk menentukan *ground truth* yang nantinya digunakan pada proses evaluasi.

### Jaccard Similarity (Ground Truth)
Jaccard Similarity untuk Kasus *Binary* [[4](Hal 4-5)](https://link.springer.com/article/10.1007/s13278-020-00660-9):
$$\text{S}_{\text{Jaccard}}(\vec{x}, \vec{y})=\frac{\text{N}_{11}}{\text{N}_{10}+\text{N}_{01}+\text{N}_{11}}$$
  Dengan:
  * $\text{N}_{11}$ adalah banyaknya elemen dengan index bersesuaian yang bernilai $1$ pada $x$ dan $y$.
  * $\text{N}_{10}$ adalah banyaknya elemen dengan index bersesuaian yang bernilai $1$ pada $x$ dan bernilai $0$ pada $y$.
  * $\text{N}_{01}$ adalah banyaknya elemen dengan index bersesuaian yang bernilai $0$ pada $x$ dan bernilai $1$ pada $y$.

Kelebihan utama metode Jaccard Similarity adalah menggunakan sudut pandang vektor sebagai himpunan yang cocok digunakan untuk sistem rekomendasi film (*content based filtering* - *by genre*). Metode ini dapat menangani dua film dengan banyak genre yang berbeda. Tingkat kemiripan dua film pada metode ini dipengaruhi oleh banyaknya elemen / genre yang sama (relevan). Semakin banyak genre yang relevan maka tingkat kemiripan akan meningkat. 

Namun tidak hanya itu, banyak genre lain yang berbeda (tidak relevan) juga mempunyai pengaruh untuk menurunkan kemiripan antara dua film tersebut. Sehingga tercipta keseimbangan tingkat kemiripan yang "ideal" antara dua film. Hal tersebutlah yang menjadikan metode ini dipilih sebagai *ground truth* pada kasus ini.

Sedangkan kekurangan dari metode ini adalah lamanya waktu komputasi untuk menangani data yang besar. Mengingat metode ini melakukan komputasi pada setiap elemen di vektor atau himpunan data.

Untuk implementasinya dengan membuat fungsi baru `jaccard_handler` mengisi nilai parameter `metric` pada fungsi `pairwise_distance` di *library* sklearn. Kemudian digunakan JIT Compiler dari *library* numba untuk optimalisasi kecepatan eksekusi kode. Berikut *output* yang dihasilkan:
```
Execution Time Jaccard Similarity (Seconds) : 101.19560837745667
```
Dari hasil *output* di atas terlihat bahwa waktu komputasi masih lebih dari 1 menit (dengan Google Colab w/o GPU) untuk sekitar $9737 \times 9737 \approx 94 \times 10^6$ *entry data* saat ini. Hal tersebut tentu tentu kurang sesuai jika data terus mengalami pertumbuhan. Oleh karena itu, perlu dicari metode lain yang lebih baik. Namun sebelumnya akan dilakukan pengecekan terlebih dahulu hasil dari metode *Jaccard Similarity* sebagai berikut:

Tabel 10. Matrik *Jaccard Similarity (Ground Truth)*
| | Toy Story (1995) | Jumanji (1995) |	Grumpier Old Men (1995) |	Waiting to Exhale (1995) | Father of the Bride Part II (1995) |	...	| Black Butler: Book of the Atlantic (2017) |	No Game No Life: Zero (2017) | Flint (2017) | Bungo Stray Dogs: Dead Apple (2018) | Andrew Dice Clay: Dice Rules (1991) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Story (1995) | 1.000000 | 0.600000 | 0.166667 | 0.142857 | 0.200000 | ... | 0.500000 | 0.600000 | 0.000000 | 0.166667 | 0.200000 |
| Jumanji (1995) | 0.600000 |	1.000000 | 0.000000 |	0.000000 | 0.000000 |	... | 0.166667 | 0.200000 | 0.000000 | 0.000000 | 0.000000 |
| Grumpier Old Men (1995) | 0.166667 | 0.000000 | 1.000000 | 0.666667 | 0.500000 | ... | 0.200000 | 0.250000 | 0.000000 | 0.000000 | 0.500000 |
| Waiting to Exhale (1995) | 0.142857 | 0.000000 | 0.666667 | 1.000000 | 0.333333 |	... |	0.166667 |	0.200000 |	0.333333 |	0.000000 |	0.333333 |
| Father of the Bride Part II (1995) |	0.200000 |	0.000000 |	0.500000 |	0.333333 |	1.000000 |	... |	0.250000 |	0.333333 |	0.000000 |	0.000000 |	1.000000 |
| ... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |
| Black Butler: Book of the Atlantic (2017) |	0.500000 |	0.166667 |	0.200000 |	0.166667 |	0.250000 |	... |	1.000000 |	0.750000 |	0.000000 |	0.500000 |	0.250000 |
| No Game No Life: Zero (2017) |	0.600000 |	0.200000 |	0.250000 |	0.200000 |	0.333333 |	... |	0.750000 |	1.000000 |	0.000000 |	0.250000 |	0.333333 |
| Flint (2017) |	0.000000 |	0.000000 |	0.000000 |	0.333333 |	0.000000 |	...	| 0.000000 |	0.000000 |	1.000000 |	0.000000 |	0.000000 |
| Bungo Stray Dogs: Dead Apple (2018) |	0.166667 |	0.000000 |	0.000000 |	0.000000 |	0.000000 |	... |	0.500000 |	0.250000 |	0.000000 |	1.000000 |	0.000000 |
| Andrew Dice Clay: Dice Rules (1991) |	0.200000 |	0.000000 |	0.500000 |	0.333333 |	1.000000 |	... |	0.250000 |	0.333333 |	0.000000 |	0.000000 |	1.000000 |

Dari matrik *Jaccard Similarity* pada Tabel 10. di atas, didapat bahwa matrik tersebut berukuran 9737 x 9737 dengan setiap elemennya mempunyai nilai antara 0 sampai 1. Jika nilai elemennya semakin mendekati nilai 1 maka tingkat kemiripannya semakin tinggi. Contoh film Jumanji (1995) dan No Game No Life: Zero (2017) teridentifikasi mirip dengan film Toy Story (1995) dengan skor kemiripan / relevansi sebesar 0.6. Contoh lain No Game No Life: Zero (2017) mirip dengan Black Butler: Book of the Atlantic (2017) dengan skor kemiripan 0.75.

### Cosine Similarity
Rumus *Cosine Similarity* didefinisikan sebagai berikut [[5]](http://www.snet.tu-berlin.de/fileadmin/fg220/courses/SS11/snet-project/recommender-systems_asanov.pdf):
$$\text{S}_{\text{Cosinus}}(\vec{x},\vec{y})=\cos(\theta)=\frac{\vec{x}\cdot\vec{y}}{\|\vec{x}\|\|\vec{y}\|}$$
Dengan:
* $\theta$ merupakan sudut antara vektor $\vec{x}$ dan $\vec{y}$.
* $\|\vec{x}\|$ dan $\|\vec{y}\|$ berturut-turut adalah besar atau panjang vektor $\vec{x}$ dan $\vec{y}$. Dengan rumus $\|\vec{x}\|=\sqrt{\sum_{i=1}^n{x_{i}^2}}$ dimana $x_{i}$ merupakan elemen vektor $\vec{x}$ posisi ke-$i$.

Kelebihan metode ini adalah tidak bergantung pada besarnya vektor. Contoh vektor $\vec{x}=[2, 0, 4]$ dengan vektor $\vec{y}=[1, 0, 2]$ yang memiliki arah yang sama namun berbeda besarannya (akibat berbeda nilai pada fiturnya). Jika vektor $\vec{x}$ dan $\vec{y}$ dihitung tingkat kemiripan atau relevansi dengan metode ini maka nilainya $1$ (kemiripan penuh). Namun kelebihan ini dapat menjadi kekurangan jika pada kasus tertentu, makna frekuensi kemunculan fitur menjadi penting. Sedangkan pada kasus ini, *Cosine Similarity* aman digunakan karena frekuensi tiap genre pada film mempunyai bobot yang sama yaitu 0 (tidak ada) atau 1 (ada).

Untuk implementasinya menggunakan fungsi `cosine_similarity()` dari *library* sklearn dengan lama waktu komputasinya sebagai berikut.
```
Execution Time Cosine Similarity (Seconds) : 0.7583773136138916
```
Dengan hasil matrik *Cosine Similarity* nya sebagai berikut.

Tabel 11. Matrik *Cosine Similarity*
| | Toy Story (1995) |	Jumanji (1995) |	Grumpier Old Men (1995) |	Waiting to Exhale (1995) |	Father of the Bride Part II (1995) |	... |	Black Butler: Book of the Atlantic (2017) |	No Game No Life: Zero (2017) |	Flint (2017) |	Bungo Stray Dogs: Dead Apple (2018) |	Andrew Dice Clay: Dice Rules (1991) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Toy Story (1995) |	1.000000 |	0.774597 |	0.316228 |	0.258199 |	0.447214	| ... |	0.670820 |	0.774597 |	0.00000 |	0.316228 |	0.447214 |
| Jumanji (1995) |	0.774597 |	1.000000 |	0.000000 |	0.000000 |	0.000000 |	... |	0.288675 |	0.333333 |	0.00000 |	0.000000 |	0.000000 |
| Grumpier Old Men (1995) |	0.316228 |	0.000000 |	1.000000	| 0.816497 |	0.707107 |	... |	0.353553 |	0.408248 |	0.00000 |	0.000000 |	0.707107 |
| Waiting to Exhale (1995) |	0.258199 |	0.000000 |	0.816497 |	1.000000 |	0.577350 |	... |	0.288675 |	0.333333 |	0.57735 |	0.000000 |	0.577350 |
| Father of the Bride Part II (1995) |	0.447214 |	0.000000 |	0.707107 |	0.577350 |	1.000000 |	... |	0.500000 |	0.577350 |	0.00000 |	0.000000 |	1.000000 |
| ... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |
| Black Butler: Book of the Atlantic (2017) |	0.670820 |	0.288675 |	0.353553 |	0.288675 |	0.500000 |	... |	1.000000 |	0.866025 |	0.00000	| 0.707107 |	0.500000 |
| No Game No Life: Zero (2017) |	0.774597 |	0.333333 |	0.408248 |	0.333333 |	0.577350 |	... |	0.866025 |	1.000000 |	0.00000 |	0.408248 |	0.577350 |
| Flint (2017) |	0.000000 |	0.000000 |	0.000000 |	0.577350 |	0.000000 |	... |	0.000000 |	0.000000 |	1.00000 |	0.000000 |	0.000000 |
| Bungo Stray Dogs: Dead Apple (2018) |	0.316228 |	0.000000 |	0.000000 |	0.000000 |	0.000000 |	... |	0.707107 |	0.408248 |	0.00000 |	1.000000 |	0.000000 |
| Andrew Dice Clay: Dice Rules (1991) |	0.447214 |	0.000000 |	0.707107 |	0.577350 |	1.000000 |	... | 0.500000 |	0.577350 |	0.00000 |	0.000000 |	1.000000 |

Dengan metode *cosine similarity*, dihasilkan matrik korelasi antar film. Matriks tersebut berukuran 9737 film x 9737 film (masing-masing dalam sumbu X dan Y).

Dari Tabel 11. di atas, diketahui bahwa setiap elemen mempunyai *range* 0 sampai 1, semakin mendekati 1 artinya kemiripannya semakin tinggi dan berlaku sebaliknya. *Range* pada nilai fungsi cosinus sebenarnya dari -1 sampai 1, namun karena elemen-elemen (genre) pada tiap film bernilai lebih dari sama dengan nol (0 atau 1). Akibatnya nilai cosinusnya dari 0 sampai 1.

Sebagai contoh, film Waiting to Exhale (1995) dan Father of the Bride Part II (1995) teridentifikasi cukup mirip dengan film Grumpier Old Men (1995) dengan skor kemiripan lebih dari 0.7. Contoh lain, film Father of the Bride Part II (1995) teridentifikasi mirip dengan film Andrew Dice Clay: Dice Rules (1991) dengan skor kemiripan penuh (1).

### Euclidean Similarity
*Euclidean Distances* didefinisikan sebagai berikut:
$$\text{D}_{\text{Euclidean}}(\vec{x},\vec{y})=\sqrt{\sum^{n}_{i=1}(x_{i}-y_{i})^2}$$
Dengan $x_{i}$ dan $y_{i}$ berturut-turut adalah elemen ke-$i$ pada vektor $\vec{x}$ dan $\vec{y}$. 

Dalam buku berjudul `Programming Collective Intelligence [Page 10]` karya Toby Segaran (2007) [[6]](https://www.oreilly.com/library/view/programming-collective-intelligence/9780596529321/) mendefinisikan persamaan *Euclidean Similarity* sebagai berikut:
$$\text{S}_{\text{Euclidean}}(\vec{x},\vec{y})=\frac{1}{1+\text{D}_{\text{Euclidean}}(\vec{x}, \vec{y})}$$

Kelebihan Euclidean adalah dapat memperoleh nilai perbedaan antara dua vektor yang sama arahnya namun beda besarnya. Contoh vektor $\vec{x}=[2, 0, 4]$ dengan vektor $\vec{y}=[1, 0, 2]$ jika menggunakan algoritma *Cosine Similarity* maka didapat kedua vektor tersebut memiliki kesamaan penuh ($1$). Namun, jika menggunakan metode ini maka didapat perbedaan jarak sebesar:
$$\begin{align*}\text{D}_{\text{Euclidean}}(\vec{x},\vec{y})&=\sqrt{\sum^{n}_{i=1}(x_{i}-y_{i})^2}\\&=\sqrt{\sum^{3}_{i=1}(x_{i}-y_{i})^2}\\&=\sqrt{(2-1)^2 + (0-0)^2 + (4-2)^2}\\&=\sqrt{5}\approx2.236\end{align*}$$

Kemudian tingkat kemiripannya sebesar:

$$\begin{align*}\text{S}_{\text{Euclidean}}(\vec{x},\vec{y})&=\frac{1}{1+\text{D}_{\text{Euclidean}}(\vec{x}, \vec{y})}\\&=\frac{1}{1+2.236}\\&=0.309\end{align*}$$

Sedangkan kekurangan algoritma ini adalah fitur dengan frekuensi kemunculan paling banyak akan mendominasi fitur lain dalam hasil komputasi jarak euclideannya. Contoh vektor $\vec{u}=[10,2]$ dan $\vec{v}=[1, 1]$ dengan hasil jarak euclideannya didominasi oleh elemen $\vec{u}_{1}=10$ dan $\vec{v}_{1}=1$. Hal tersebut dapat diatasi dengan melakukan normalisasi atau standariasi pada data numerik [[7]](https://dl.acm.org/doi/pdf/10.1145/331499.331504). Pada kasus ini telah dilakukan normalisasi dengan *One-Hot Encoding* pada data kategorikal (genre) untuk menyamakan bobot setiap genre (frekuensi = ada tidaknya genre).

Untuk implementasinya menggunakan fungsi `euclidean_distances()` dari *library* sklearn dengan lama waktu komputasinya sebagai berikut.
```
Execution Time Euclidean Similarity (Seconds) : 1.784877061843872
```
Dengan hasil matrik *Euclidean Similarity* nya sebagai berikut.

Tabel 12. Matrik *Euclidean Similarity*
| | Toy Story (1995) |	Jumanji (1995) |	Grumpier Old Men (1995) |	Waiting to Exhale (1995) |	Father of the Bride Part II (1995) |	... |	Black Butler: Book of the Atlantic (2017) |	No Game No Life: Zero (2017) |	Flint (2017) |	Bungo Stray Dogs: Dead Apple (2018) |	Andrew Dice Clay: Dice Rules (1991) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Toy Story (1995) |	1.000000 |	0.414214 |	0.309017 |	0.289898 |	0.333333 |	... |	0.366025 |	0.414214 |	0.289898 |	0.309017 |	0.333333 |
| Jumanji (1995) |	0.414214 |	1.000000 |	0.309017 |	0.289898 |	0.333333 |	... |	0.309017 |	0.333333 |	0.333333 |	0.309017 |	0.333333 |
| Grumpier Old Men (1995) |	0.309017 |	0.309017 |	1.000000 |	0.500000 |	0.500000 |	... |	0.333333 |	0.366025 |	0.366025 |	0.333333 |	0.500000 |
| Waiting to Exhale (1995) |	0.289898 |	0.289898 |	0.500000 |	1.000000 |	0.414214 |	... |	0.309017 |	0.333333 |	0.414214 |	0.309017 |	0.414214 |
| Father of the Bride Part II (1995) |	0.333333 |	0.333333 |	0.500000 |	0.414214 |	1.000000 |	... |	0.366025 |	0.414214 |	0.414214 |	0.366025 |	1.000000 |
| ... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |	... |
| Black Butler: Book of the Atlantic (2017) |	0.366025 |	0.309017 |	0.333333 |	0.309017 |	0.366025 |	... |	1.000000 |	0.500000 |	0.309017 |	0.414214 |	0.366025 |
| No Game No Life: Zero (2017) |	0.414214 |	0.333333 |	0.366025 |	0.333333 |	0.414214 |	... |	0.500000 |	1.000000 |	0.333333 |	0.366025 |	0.414214 |
| Flint (2017) |	0.289898 |	0.333333 |	0.366025 |	0.414214 |	0.414214 |	... |	0.309017 |	0.333333 |	1.000000 |	0.366025 |	0.414214 |
| Bungo Stray Dogs: Dead Apple (2018) |	0.309017 |	0.309017 |	0.333333 |	0.309017 |	0.366025 |	... |	0.414214 |	0.366025 |	0.366025 |	1.000000 |	0.366025 |
| Andrew Dice Clay: Dice Rules (1991) |	0.333333 |	0.333333 |	0.500000 |	0.414214 |	1.000000 |	... |	0.366025 |	0.414214 |	0.414214 |	0.366025 |	1.000000 |

Hasil *Euclidean Distances* ditransformasikan ke dalam skala 0 sampai 1 dengan persamaan *Euclidean Similarity*. Jika nilai elemen matriks mendekati 1 artinya kemiripannya semakin tinggi.

Sebagai contoh, pada Tabel 12. diatas didapat bahwa film Father of the Bride Part II (1995) teridentifikasi mirip dengan film Andrew Dice Clay: Dice Rules (1991) dengan skor kemiripan penuh yaitu (1).

Kemudian jika dilakukan pengecekan persebaran nilai pada matriks tersebut, diperoleh:

Tabel 13. Daftar *Unique Value* pada Elemen Matrik *Euclidean Similarity*
| *Unique Value* |
|---|
| 0.2 |
| 0.2052131 |
| 0.21089672 |
| 0.21712927 |
| 0.22400924 |
| 0.23166248 |
| 0.24025307 |
| 0.25 |
| 0.26120387 |
| 0.27429189 |
| 0.28989795 |
| 0.30901699 |
| 0.33333333 |
| 0.3660254 |
| 0.41421356 |
| 0.5 |
| 1. |

Dari hasil *output* di atas, menunjukkan bahwa skor kemiripan tertinggi kedua adalah 0.5. Hal ini dikarenakan ada tepat satu genre yang berbeda pada dua film yang diperbandingkan. Sehingga nilai *Euclidean Distance* nya sama dengan $1$ dan *Euclidean Similarity* nya sama dengan $\frac{1}{1+1}=0.5$.

Contohnya film Waiting to Exhale (1995) dan Father of the Bride Part II (1995) teridentifikasi cukup mirip (berbeda satu genre) dengan film Grumpier Old Men (1995) dengan skor kemiripan sama dengan 0.5 yang dapat dilihat pada Tabel 12.

Tabel 14. Contoh Film dengan Tingkat Kemiripan 0.5

|index|movieId|title|genres|
|---|---|---|---|
|2|3|Grumpier Old Men \(1995\)|Comedy&#124;Romance|
|3|4|Waiting to Exhale \(1995\)|Comedy&#124;Drama&#124;Romance|
|4|5|Father of the Bride Part II \(1995\)|Comedy|

Dari Tabel 14. terlihat pasangan film (Grumpier Old Men, Waiting to Exhale) memiliki dua genre yang sama (*Comedy & Romance*) dan satu genre yang berbeda (*Drama*). Sedangkan pasangan film (Grumpier Old Men, Father of the Bride Part II) hanya memiliki satu genre yang sama (*Comedy*) dan satu genre yang berbeda (*Romance*).
Namun kedua pasangan film tersebut mempunyai kesamaan nilai kemiripan sebesar 0.5.

## Evaluation

Setelah mendapatkan matrik korelasi antar film dengan metode *Cosine Similarity* dan *Euclidean Similarity* selanjutnya adalah membangun sistem rekomendasi berdasarkan matrik korelasi yang telah dibuat. Namun, sebelum itu perlu ditentukan terlebih dahulu sampel yang akan digunakan untuk mengevaluasi performa setiap metode.

### Sampel Film untuk Evaluasi
Untuk pemilihan sampelnya akan dipilih secara acak untuk setiap banyak genre yang berbeda pada dataset. Dari Gambar 2. (Banyak Genre Tiap Film), diketahui bahwa minimal banyak genre adalah 1 dan maksimalnya adalah 10. Sehingga untuk banyak genre 1,2,3,4,5,6,7,8,10 masing-masing akan diambil secara acak satu sampel yang nantinya akan diperoleh total sembilan sampel. Penentuan sampel dengan aturan tersebut menjadikan distribusi sampel merata yang nantinya akan digunakan pada proses perhitungan skor metrik. Berikut daftar sampel yang terpilih secara acak.

Tabel 15. Daftar Sampel
|index|movieId|title|genres|num\_genre|
|---|---|---|---|---|
|4465|6592|Secret Lives of Dentists, The \(2002\)|Drama|1|
|303|345|Adventures of Priscilla, Queen of the Desert, The \(1994\)|Comedy&#124;Drama|2|
|8147|102070|Grabbers \(2012\)|Comedy&#124;Horror&#124;Sci-Fi|3|
|9697|184987|A Wrinkle in Time \(2018\)|Adventure&#124;Children&#124;Fantasy&#124;Sci-Fi|4|
|6486|53121|Shrek the Third \(2007\)|Adventure&#124;Animation&#124;Children&#124;Comedy&#124;Fantasy|5|
|5516|26504|Cloak & Dagger \(1984\)|Action&#124;Adventure&#124;Children&#124;Crime&#124;Mystery&#124;Thriller|6|
|6626|56152|Enchanted \(2007\)|Adventure&#124;Animation&#124;Children&#124;Comedy&#124;Fantasy&#124;Musical&#124;Romance|7|
|5556|26701|Patlabor: The Movie \(Kidô keisatsu patorebâ: The Movie\) \(1989\)|Action&#124;Animation&#124;Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Sci-Fi&#124;Thriller|8|
|7441|81132|Rubber \(2010\)|Action&#124;Adventure&#124;Comedy&#124;Crime&#124;Drama&#124;Film-Noir&#124;Horror&#124;Mystery&#124;Thriller&#124;Western|10|

Untuk tahapan selanjutnya (menampilkan top-k rekomendasi), akan menggunakan salah satu sampel dari kesembilan sampel pada tabel di atas. Tujuannya sebagai salah satu contoh top-k hasil rekomendasi dan meringkas penulisan supaya tidak terkesan terlalu panjang terutama jika nilai k nya cukup besar seperti top-100 atau lebih. Sedangkan pada tahap evaluasi model dengan metrik $\text{nDCG@}k$ akan menggunakan semua sampel pada tabel di atas.

### Top-k Rekomedasi
Pada tahap ini akan diperlihatkan hasil top-k rekomendasi film yang relevan dengan sampel film. Sampel film yang dipilih adalah `Patlabor: The Movie (Kidô keisatsu patorebâ: The Movie) (1989)`. Untuk pemilihannya sendiri tidak ada aturan tertentu pada kesembilan sampel sebelumnya. Mengingat tujuan tahap top-k rekomendasi ini masih hanya memperlihatkan contoh hasil top-k rekomendasi film. Sedangkan untuk pengujian performanya akan ditentukan pada tahap perhitungan skor metrik $\text{nDCG@}k$ (bagian selanjutnya).

Untuk implementasinya, akan menggunakan nilai k sebesar 50 atau menampilkan top-50 rekomendasi film.

#### Dengan Jaccard Similarity (Ground Truth)
Tabel 16. Top-50 Rekomendasi dengan *Jaccard Similarity* (*Ground Truth*)
|index|title|movieId|genres|JSS|
|---|---|---|---|---|
|0|Strange Days \(1995\)|198|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.75|
|1|Inception \(2010\)|79132|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller&#124;IMAX|0\.667|
|2|Sherlock: The Abominable Bride \(2016\)|150548|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|3|RoboCop \(1987\)|2985|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.625|
|4|Sin City \(2005\)|32587|Action&#124;Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.625|
|5|Scanner Darkly, A \(2006\)|27904|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|6|Cellular \(2004\)|8860|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|7|Batman Beyond: Return of the Joker \(2000\)|27311|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.625|
|8|Negotiator, The \(1998\)|2058|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|9|Ghost in the Shell: Solid State Society \(2006\)|151781|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.625|
|10|Insomnia \(2002\)|5388|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|11|Knowing \(2009\)|67197|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|12|Minority Report \(2002\)|5445|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|13|Renaissance \(2006\)|44849|Action&#124;Animation&#124;Film-Noir&#124;Sci-Fi&#124;Thriller|0\.625|
|14|Blackhat \(2015\)|120637|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|15|Girl Who Played with Fire, The \(Flickan som lekte med elden\) \(2009\)|74510|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|16|Source Code \(2011\)|85414|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|17|X-Files: Fight the Future, The \(1998\)|1909|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|18|Kite \(2014\)|132618|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|19|Ghost in the Shell 2: Innocence \(a\.k\.a\. Innocence\) \(Inosensu\) \(2004\)|27728|Action&#124;Animation&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.625|
|20|Double, The \(2011\)|90738|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|21|Man on Fire \(2004\)|7445|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|22|Balance \(1989\)|72104|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.625|
|23|Too Late for Tears \(1949\)|130482|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.625|
|24|Jack Reacher: Never Go Back \(2016\)|165347|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.625|
|25|RoboCop 3 \(1993\)|519|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.625|
|26|Mulholland Drive \(2001\)|4848|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.625|
|27|Watchmen \(2009\)|60684|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller&#124;IMAX|0\.556|
|28|Shaft \(1971\)|3729|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|29|Omega Man, The \(1971\)|3032|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.5|
|30|General's Daughter, The \(1999\)|2688|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.5|
|31|Munich \(2005\)|41997|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|32|Point Blank \(1967\)|26172|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|33|Cowboy Bebop: The Movie \(Cowboy Bebop: Tengoku no Tobira\) \(2001\)|6283|Action&#124;Animation&#124;Sci-Fi&#124;Thriller|0\.5|
|34|Man Apart, A \(2003\)|6280|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|35|Passion \(2012\)|103449|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.5|
|36|Killer, The \(Die xue shuang xiong\) \(1989\)|1218|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|37|Wonderland \(2003\)|6868|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.5|
|38|House of Games \(1987\)|4037|Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.5|
|39|Evangelion: 2\.0 You Can \(Not\) Advance \(Evangerion shin gekijôban: Ha\) \(2009\)|84187|Action&#124;Animation&#124;Drama&#124;Sci-Fi|0\.5|
|40|Confidential Report \(1955\)|26002|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.5|
|41|Chinatown \(1974\)|1252|Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.5|
|42|Dark Blue \(2003\)|6185|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|43|Brick \(2005\)|44761|Crime&#124;Drama&#124;Film-Noir&#124;Mystery|0\.5|
|44|Face/Off \(1997\)|1573|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|45|Pledge, The \(2001\)|4056|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.5|
|46|RoboCop 2 \(1990\)|2986|Action&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.5|
|47|Blood In, Blood Out \(1993\)|3761|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|
|48|Strangers on a Train \(1951\)|2186|Crime&#124;Drama&#124;Film-Noir&#124;Thriller|0\.5|
|49|Cradle 2 the Grave \(2003\)|6196|Action&#124;Crime&#124;Drama&#124;Thriller|0\.5|

Dari Tabel 16. di atas, menunjukan top-50 dari metode *Jaccard Similarity* yang menjadi *ground truth* pada kasus ini. JSS adalah singkatan dari *Jaccard Similarity Score* yang didapat pada matrik *Jaccard Similarity* (Tabel 10) dan menjadi patokan dalam menentukan urutan hasil rekomendasi. Perlu diperhatikan bahwa peringkat 3 sampai 27 memiliki skor yang sama yaitu 0.625 (kemiripan berdasarkan genre antara sampel dengan hasil rekomendasi). Di sisi lain karena bobot setiap film ditentukan oleh genre yang dimilikinya, maka film peringkat 3 sampai 27 memiliki bobot yang sama. Artinya untuk peringkat 3 dapat ditempati film Sherlock: The Abominable Bride (2016), RoboCop (1987), Sin City (2005) atau film lain dalam peringkat 3 sampai 27. Hal ini tentu akan membingungkan untuk menentukan urutan ideal top-k rekomendasi sebagai *ground truth* pada proses evaluasi.

Oleh karena itu, urutan ideal pada top-k rekomendasi akan didasari pada urutan besaran skor pada JSS. Idealnya skor harus urut dari skor JSS tertinggi hingga terendah seperti pada Tabel 16. (top-k rekomendasi dengan *Jaccard Similarity*) di atas.

#### Dengan Cosine Similarity
Tabel 17. Top-50 Rekomendasi dengan *Cosine Similarity*
|index|title|movieId|genres|CSS|JSS|
|---|---|---|---|---|---|
|0|Strange Days \(1995\)|198|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.8660254037844387|0\.75|
|1|Inception \(2010\)|79132|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller&#124;IMAX|0\.801783725737273|0\.667|
|2|Batman Beyond: Return of the Joker \(2000\)|27311|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|3|Ghost in the Shell 2: Innocence \(a\.k\.a\. Innocence\) \(Inosensu\) \(2004\)|27728|Action&#124;Animation&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|4|Negotiator, The \(1998\)|2058|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|5|Scanner Darkly, A \(2006\)|27904|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|6|Kite \(2014\)|132618|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|7|Mulholland Drive \(2001\)|4848|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|8|Cellular \(2004\)|8860|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|9|Sin City \(2005\)|32587|Action&#124;Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|10|Source Code \(2011\)|85414|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|11|X-Files: Fight the Future, The \(1998\)|1909|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|12|Jack Reacher: Never Go Back \(2016\)|165347|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|13|Too Late for Tears \(1949\)|130482|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|14|RoboCop \(1987\)|2985|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|15|Renaissance \(2006\)|44849|Action&#124;Animation&#124;Film-Noir&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|16|Girl Who Played with Fire, The \(Flickan som lekte med elden\) \(2009\)|74510|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|17|Balance \(1989\)|72104|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|18|Insomnia \(2002\)|5388|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|19|Minority Report \(2002\)|5445|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|20|Knowing \(2009\)|67197|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|21|RoboCop 3 \(1993\)|519|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|22|Double, The \(2011\)|90738|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|23|Sherlock: The Abominable Bride \(2016\)|150548|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|24|Man on Fire \(2004\)|7445|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|25|Blackhat \(2015\)|120637|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7905694150420948|0\.625|
|26|Ghost in the Shell: Solid State Society \(2006\)|151781|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.7905694150420948|0\.625|
|27|Watchmen \(2009\)|60684|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller&#124;IMAX|0\.7216878364870323|0\.556|
|28|Hostage \(2005\)|32029|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|29|The Fate of the Furious \(2017\)|170875|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|30|Soylent Green \(1973\)|2009|Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|31|Double Jeopardy \(1999\)|2881|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|32|Doomsday \(2008\)|58297|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|33|Tokyo Tribe \(2014\)|138632|Action&#124;Crime&#124;Drama&#124;Sci-Fi|0\.7071067811865475|0\.5|
|34|Elite Squad \(Tropa de Elite\) \(2007\)|55721|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|35|Abduction \(2011\)|90524|Action&#124;Drama&#124;Mystery&#124;Thriller|0\.7071067811865475|0\.5|
|36|L\.A\. Confidential \(1997\)|1617|Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.7071067811865475|0\.5|
|37|Donnie Darko \(2001\)|4878|Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|38|Contraband \(2012\)|91842|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|39|Mesrine: Killer Instinct \(L'instinct de mort\) \(2008\)|65596|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|40|Rise of the Planet of the Apes \(2011\)|88744|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|41|Omega Man, The \(1971\)|3032|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|42|Evangelion: 2\.0 You Can \(Not\) Advance \(Evangerion shin gekijôban: Ha\) \(2009\)|84187|Action&#124;Animation&#124;Drama&#124;Sci-Fi|0\.7071067811865475|0\.5|
|43|History of Violence, A \(2005\)|37733|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|44|Clockwork Orange, A \(1971\)|1206|Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|45|Unleashed \(Danny the Dog\) \(2005\)|33437|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|
|46|Ghost in the Shell Arise - Border 2: Ghost Whispers \(2013\)|139859|Action&#124;Animation&#124;Sci-Fi&#124;Thriller|0\.7071067811865475|0\.5|
|47|Devil in a Blue Dress \(1995\)|164|Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.7071067811865475|0\.5|
|48|Every Secret Thing \(2014\)|140850|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.7071067811865475|0\.5|
|49|Spy Game \(2001\)|4901|Action&#124;Crime&#124;Drama&#124;Thriller|0\.7071067811865475|0\.5|

Pada hasil top-50 rekomendasi (Tabel 17) di atas, kolom CSS adalah *Cosine Similarity Score* yang jika dibandingkan dengan kolom JSS terlihat memiliki urutan yang sama dari besar ke kecil tanpa ada yang *miss* dalam pengurutannya. Untuk memastikannya kembali akan dilakukan perhitungan dengan metrik $\text{nDCG@}k$ (bagian selanjutnya).

#### Dengan Euclidean Similarity
Tabel 18. Top-50 Rekomendasi dengan *Cosine Similarity*
|index|title|movieId|genres|ESS|JSS|
|---|---|---|---|---|---|
|0|Strange Days \(1995\)|198|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.4142135623730951|0\.75|
|1|Girl Who Played with Fire, The \(Flickan som lekte med elden\) \(2009\)|74510|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|2|Too Late for Tears \(1949\)|130482|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|3|Scanner Darkly, A \(2006\)|27904|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|4|Minority Report \(2002\)|5445|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|5|RoboCop \(1987\)|2985|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|6|Double, The \(2011\)|90738|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|7|Man on Fire \(2004\)|7445|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|8|Ghost in the Shell: Solid State Society \(2006\)|151781|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|9|Mulholland Drive \(2001\)|4848|Crime&#124;Drama&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|10|Balance \(1989\)|72104|Animation&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|11|Cellular \(2004\)|8860|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|12|Batman Beyond: Return of the Joker \(2000\)|27311|Action&#124;Animation&#124;Crime&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|13|X-Files: Fight the Future, The \(1998\)|1909|Action&#124;Crime&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|14|Jack Reacher: Never Go Back \(2016\)|165347|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|15|Blackhat \(2015\)|120637|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|16|Ghost in the Shell 2: Innocence \(a\.k\.a\. Innocence\) \(Inosensu\) \(2004\)|27728|Action&#124;Animation&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|17|Inception \(2010\)|79132|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller&#124;IMAX|0\.36602540378443865|0\.667|
|18|Sin City \(2005\)|32587|Action&#124;Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|19|Sherlock: The Abominable Bride \(2016\)|150548|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|20|Source Code \(2011\)|85414|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|21|Renaissance \(2006\)|44849|Action&#124;Animation&#124;Film-Noir&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|22|Knowing \(2009\)|67197|Action&#124;Drama&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|23|Negotiator, The \(1998\)|2058|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|24|Kite \(2014\)|132618|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|25|Insomnia \(2002\)|5388|Action&#124;Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.36602540378443865|0\.625|
|26|RoboCop 3 \(1993\)|519|Action&#124;Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.36602540378443865|0\.625|
|27|Cowboy Bebop: The Movie \(Cowboy Bebop: Tengoku no Tobira\) \(2001\)|6283|Action&#124;Animation&#124;Sci-Fi&#124;Thriller|0\.3333333333333333|0\.5|
|28|Getaway, The \(1972\)|8016|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|29|Irreversible \(Irréversible\) \(2002\)|6214|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|30|Tokyo Tribe \(2014\)|138632|Action&#124;Crime&#124;Drama&#124;Sci-Fi|0\.3333333333333333|0\.5|
|31|Sherlock Holmes \(2009\)|73017|Action&#124;Crime&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|32|Blood In, Blood Out \(1993\)|3761|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|33|Taken 2 \(2012\)|96861|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|34|Abduction \(2011\)|90524|Action&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|35|American Friend, The \(Amerikanische Freund, Der\) \(1977\)|6021|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|36|Memories of Murder \(Salinui chueok\) \(2003\)|31364|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|37|Omega Man, The \(1971\)|3032|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.3333333333333333|0\.5|
|38|Assault on Precinct 13 \(2005\)|31420|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|39|Fast and the Furious: Tokyo Drift, The \(Fast and the Furious 3, The\) \(2006\)|46335|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|40|Pledge, The \(2001\)|4056|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|41|Outbreak \(1995\)|292|Action&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.3333333333333333|0\.5|
|42|The Fate of the Furious \(2017\)|170875|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|43|Devil in a Blue Dress \(1995\)|164|Crime&#124;Film-Noir&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|
|44|Steamboy \(Suchîmubôi\) \(2004\)|31660|Action&#124;Animation&#124;Drama&#124;Sci-Fi|0\.3333333333333333|0\.5|
|45|Clockwork Orange, A \(1971\)|1206|Crime&#124;Drama&#124;Sci-Fi&#124;Thriller|0\.3333333333333333|0\.5|
|46|Self/less \(2015\)|135518|Action&#124;Mystery&#124;Sci-Fi&#124;Thriller|0\.3333333333333333|0\.5|
|47|Miami Vice \(2006\)|47044|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|48|Real McCoy, The \(1993\)|7031|Action&#124;Crime&#124;Drama&#124;Thriller|0\.3333333333333333|0\.5|
|49|La Cérémonie \(1995\)|1406|Crime&#124;Drama&#124;Mystery&#124;Thriller|0\.3333333333333333|0\.5|

Berdasarkan Tabel 18. di atas, terdapat hal menarik pada urutan ke-18 dengan film berjudul Inception (2010) pada kolom ESS (*Euclidean Similarity Score*) memiliki skor yang sama pada peringkat ke-17 dan ke-19 yaitu sebesar 0.366025. Sedangkan pada kolom JSS memiliki skor 0.666667 yang berbeda dengan peringkat ke-17 dan ke-19 yang memiliki skor 0.625. Hal ini menunjukkan adanya tingkat *error* pada hasil rekomendasi terhadap *ground truth*. Untuk mengevaluasinya akan dilakukan dengan metrik $\text{nDCG@}k$ sebagai berikut.

### Skor Metrik (nDCG@k)
Metrik *Normalized Discounted Cumulative Gain* (nDCG) dipilih karena mampu mengevaluasi model dengan tingkat kemiripan / relevansi yang beragam. Contoh film A dan B tingkat relevansi 0 (tidak relevan semua genrenya), A dan C sebesar 1 (sebagian genre relevan), A dan D sebesar 2 (sangat relevan genrenya). Tingkat relevansi pada kasus ini didasari pada metode yang digunakan (*cosine similarity* atau *euclidean similarity*).

Awal mula perhitungannya menggunakan *Cumulative Gain* (CG) sebagai berikut [[8]](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf):
$$\text{CG}[k] = \left\{{\begin{array}{lc}\text{G}[1], & \text{jika }k=1\\\text{CG}[k-1] + \text{G}[k], & \text{jika } k\neq 1\end{array}}\right.$$
Atau dapat ditulis ulang untuk top-k rekomendasi sebagai berikut:
$$\text{CG@}k = \sum^{k}_{i=1}{\text{G}[i]}$$
Dengan:
* $\text{CG}[k]$ atau $\text{CG@}k$ adalah *Cumulative Gain* pada top-k rekomendasi
* $\text{G}[i]$ adalah *Gain* atau tingkat relevansi pada hasil rekomendasi urutan ke-$i$

*Discounted Cumulated Gain* (DCG) didefinisikan sebagai berikut [[8]](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf):

$$\text{DCG}[k]=\left\{{\begin{array}{lc}\text{CG}[k], & \text{jika }k < b\\\text{DCG}[k-1] + \dfrac{\text{G}[k]}{^{b}\log{k}}, & \text{jika } k\geq b\end{array}}\right.$$

Dipilih $b=2$ karena cukup *smooth* dalam melakukan *discount*. Contoh *discount* pada rekomendasi urutan ke-4 maka pembaginya $^{2}\log{4} = 2$, sedangkan pada urutan ke-1024 pembaginya $^{2}\log{1024} = 10$ (rentangnya tidak terlalu jauh). Jika ditulis ulang untuk top-k rekomendasi sebagai berikut:
$$\text{DCG@}k=\text{CG}[1] + \sum^{k}_{i=2}{\dfrac{\text{G}[i]}{^{2}\log{i}}}$$
Dengan $\text{DCG}[k]$ atau $\text{DCG@}k$ adalah *Discounted Cumulated Gain* pada top-k rekomendasi.

Muncul masalah baru dari $\text{CG@}k$ maupun $\text{DCG@}k$ yaitu nilainya akan semakin membesar jika $k$ membesar. Hal ini tentu menjadikan sulit untuk melakukan komparasi performa model pada top-k rekomendasi terutama jika berbeda nilai $k$. Untuk mengatasinya digunakan metode normalisasi untuk mengubah hasil $\text{DCG@}k$ ke dalam jangkauan 0 sampai 1.

*Normalized Discounted Cumulative Gain* (nDCG) didefinisikan sebagai berikut [[9]](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ecir2008.pdf):
$$\text{nDCG@}k=\frac{\text{DCG@}k}{\text{iDCG@}k}$$
Dengan:
* $\text{nDCG@}k$ adalah *Normalized Discounted Cumulative Gain* pada top-k rekomendasi. Semakin mendekati nilai 1 maka performa model semakin baik karena tingkat relevansinya tinggi pada hasil rekomendasinya (terhadap *ground truth*).
* $\text{iDCG@}k$ adalah *Ideal Discounted Cumulative Gain* yang merupakan nilai $\text{DCG@}k$ dari urutan *ideal* pada top-k rekomendasi. Catatan: urutan *ideal* diperoleh dari urutan top-k rekomendasi pada *ground truth*. 

Berikut hasil perhitungan metrik $\text{nDCG@}k$ pada setiap sampel.

Tabel 19. Hasil Metrik $\text{nDCG@}k$ pada Setiap Sampel
|index|movieId|title|num\_genre|nDCG@3 Cosine|nDCG@3 Euclidean|nDCG@10 Cosine|nDCG@10 Euclidean|nDCG@20 Cosine|nDCG@20 Euclidean|nDCG@50 Cosine|nDCG@50 Euclidean|
|---|---|---|---|---|---|---|---|---|---|---|---|
|4465|6592|Secret Lives of Dentists, The \(2002\)|1|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|
|303|345|Adventures of Priscilla, Queen of the Desert, The \(1994\)|2|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|
|8147|102070|Grabbers \(2012\)|3|1\.0|1\.0|1\.0|1\.0|1\.0|0\.980587|1\.0|0\.985246|
|9697|184987|A Wrinkle in Time \(2018\)|4|1\.0|1\.0|1\.0|0\.988983|1\.0|0\.996200|0\.999544|0\.984071|
|6486|53121|Shrek the Third \(2007\)|5|1\.0|1\.0|1\.0|1\.0|1\.0|0\.992273|1\.0|0\.996082|
|5516|26504|Cloak & Dagger \(1984\)|6|1\.0|0\.964745|1\.0|0\.981882|1\.0|0\.989861|1\.0|0\.959502|
|6626|56152|Enchanted \(2007\)|7|1\.0|1\.0|1\.0|0\.995039|1\.0|0\.996486|1\.0|0\.994578|
|5556|26701|Patlabor: The Movie \(Kidô keisatsu patorebâ: The Movie\) \(1989\)|8|1\.0|0\.976992|1\.0|0\.987925|1\.0|0\.993727|1\.0|0\.994732|
|7441|81132|Rubber \(2010\)|10|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|1\.0|0\.991952|

Untuk mendapatkan gambaran performa setiap metode, akan dihitung rata-rata nilai $\text{nDCG@}k$ sebagai berikut:

Tabel 20. Rata-Rata $\text{nDCG@}k$
||*Cosine Similarity*|*Euclidean Similarity*|
|---|---|---|
|*Mean* nDCG@3|1\.0|0\.9935263455222254|
|*Mean* nDCG@10|1\.0|0\.9948698273327989|
|*Mean* nDCG@20|1\.0|0\.9943481561376465|
|*Mean* nDCG@50|0\.9999493394186207|0\.9895738094588804|

Berdasarkan Tabel 20. di atas nilai $\text{nDCG@}k$ pada *Cosine Similarity* terlihat memiliki hasil rekomendasi yang sangat mirip dengan *ground truth* pada top-3, top-10 dan top-20 dengan skor 1. Sedangkan pada top-50 rekomendasi mengalami penurunan menjadi 0.99994934. Disisi lain pada metode *Euclidean Similarity* terlihat memiliki nilai $\text{nDCG@}k$ yang tinggi pada top-10 mencapai 0.99486983 dan mengalami penurunan pada top-3 (99352634), top-20 (0.99434816) dan top-50 (0.98957381).

Untuk lebih jelasnya dapat dilihat pada visualisasi berikut.

![](https://aneechan.github.io/assets/picture/mlt-s2/visualisasi-rata-rata-ndcgk.png)

Gambar 3. Visualisasi Rata-Rata $\text{nDCG@}k$

Dari visualisasi di atas, terlihat bahwa secara keseluruhan metode *Cosine Similarity* memiliki skor $\text{nDCG@}k$ yang lebih tinggi dibandingkan dengan metode *Euclidean Similarity*. Skor $\text{nDCG@}k$ terendah *Euclidean Similarity* adalah 0.98957381 yang artinya tingkat *error* terhadap *ground truth* adalah sebesar 0.01042619 atau sekitar 1.04%. Sedangkan pada *Cosine Similarity* skor terendahnya sebesar 0.99994934 yang artinya tingkat *error* terhadap *ground truth* sebesar 0.00005066 atau sekitar 0.005%.

### Komparasi Waktu Komputasi
Selain metrik $\text{nDCG@}k$ kita juga perlu mempertimbangkan lama waktu komputasi pada kedua metode yang digunakan. Berikut hasil komparasi lama waktu eksekusi tiap metode.

Tabel 21. Komparasi Waktu Komputasi
||*Jaccard Similarity \(Ground Truth\)*|*Cosine Similarity*|*Euclidean Similarity*|
|---|---|---|---|
|*Time \(Seconds\)*|101\.19560837745667|0\.7583773136138916|1\.784877061843872|

Dari hasil tabel di atas, terdapat perbedaan yang signifikan antara lama waktu eksekusi metode *Jaccard Similarity* yang digunakan pada *ground trutuh* dengan dua metode lainnya (*Cosine Similarity* dan *Euclidean Similarity*). Dimana selisih waktunya lebih dari 1 menit. Kemudian untuk waktu eksekusi metode *Cosine Similarity* lebih cepat dibandingkan waktu eksekusi metode *Euclidean Similarity*.

## Conclusion
Berdasarkan hasil evaluasi model di atas, dapat kita simpulkan bahwa model terbaik untuk melakukan rekomendasi film berdasarkan genre adalah model dengan metode Cosine Similarity. Hal ini terbukti dengan nilai $\text{nDCG@}k$ pada metode *Cosine Similarity* terbukti lebih tinggi dari metode *Euclidean Similarity* baik itu pada top-3, top-10, top-20 maupun top-50 rekomendasi film. Artinya hasil rekomendasi oleh metode *Cosine Similarity* mempunyai tingkat relevansi yang lebih tinggi dari pada metode *Euclidean Similarity*. Selain itu, waktu komputasi model dengan metode *Cosine Similarity* (0.7584 detik) terbukti lebih cepat daripada metode *Euclidean Similarity* (1.7849 detik).

Kemudian perbandingan hasil rekomendasi antara metode *Cosine Similarity* dengan metode *Jaccard Similarity (ground truth)* menunjukkan tingkat kemiripan yang tinggi. Dengan nilai $\text{nDCG@}k$ pada top-3 (skor: 1), top-10 (skor: 1), top-20 (skor: 1) dan top-50 (skor: 0.99994934). Selain itu, untuk lama waktu komputasi metode *Cosine Similarity* lebih unggul dengan perolehan waktu (0.7584 detik) yang 100 detik lebih cepat dibandingkan waktu komputasi metode *Jaccard Similarity (ground truth)* (101.1956 detik). Sehingga metode *Cosine Similarity* dapat menjadi alternatif utama pengganti *Jaccard Similarity* yang sebelumnya dijadikan sebagai *ground truth*.

## Daftar Referensi
[1] Dooms, S., Audenaert, P., Fostier, J. et al. In-memory, distributed content-based recommender system. J Intell Inf Syst 42, 645–669 (2014). https://doi.org/10.1007/s10844-013-0276-1. Tersedia: [Springer Link](https://link.springer.com/article/10.1007/s10844-013-0276-1#citeas)  
[2] F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19. https://doi.org/10.1145/2827872. Tersedia: [tautan](http://files.grouplens.org/papers/harper-tiis2015.pdf). Diakses pada November 2022.  
[3] A. Koufakou, J. Secretan, J. Reeder, K. Cardona and M. Georgiopoulos, "Fast parallel outlier detection for categorical datasets using MapReduce," 2008 IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), 2008, pp. 3298-3304, doi: 10.1109/IJCNN.2008.4634266. Tersedia: [tautan](https://www.eecs.ucf.edu/georgiopoulos/sites/default/files/247.pdf). Diakses pada Oktober 2022.  
[4] Verma, V., Aggarwal, R.K. A comparative analysis of similarity measures akin to the Jaccard index in collaborative recommendations: empirical and theoretical perspective. Soc. Netw. Anal. Min. 10, 43 (2020). https://doi.org/10.1007/s13278-020-00660-9. Tersedia: [Springer Link](https://link.springer.com/article/10.1007/s13278-020-00660-9).  
[5] Melville, Prem, and Vikas Sindhwani. "Recommender systems." Encyclopedia of machine learning 1 (2010): 829-838. Tersedia: [tautan](http://www.snet.tu-berlin.de/fileadmin/fg220/courses/SS11/snet-project/recommender-systems_asanov.pdf). Diakses pada November 2022.  
[6] Segaran, Toby. "Programming Collective Intelligence". O'Reilly Media, Inc. 2007. Tersedia: [O'Reilly Media](https://www.oreilly.com/library/view/programming-collective-intelligence/9780596529321/).  
[7] Jain, A. K.; Murty, M. N.; Flynn, P. J. (1999). Data clustering: a review. ACM Computing Surveys, 31(3), 264–323. doi:10.1145/331499.331504. Tersedia: [tautan](https://dl.acm.org/doi/pdf/10.1145/331499.331504). Diakses pada Oktober 2022.  
[8] Jarvelin, K., & Kekalainen, J. (2002). Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS), 20(4), 422-446. Tersedia: [tautan](https://faculty.cc.gatech.edu/~zha/CS8803WST/dcg.pdf). Diakses pada November 2022.  
[9] McSherry, F., & Najork, M. (2008, March). Computing information retrieval performance measures efficiently in the presence of tied scores. In European conference on information retrieval (pp. 414-421). Springer, Berlin, Heidelberg. Tersedia: [tautan](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ecir2008.pdf). Diakses pada November 2022.