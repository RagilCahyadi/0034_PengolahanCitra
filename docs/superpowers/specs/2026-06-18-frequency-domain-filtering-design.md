# Frequency Domain Filtering — Design Spec

**Tanggal:** 2026-06-18
**Notebook target:** `w10_FrequencyDomain.ipynb`
**Tujuan:** Memahami & mengimplementasikan tiga filter frequency domain baru berdasarkan logika ideal high-pass yang sudah ada.

## Latar Belakang

Notebook saat ini sudah punya alur lengkap:

```
img → fft2 → fftshift → (× filter) → ifftshift → ifft2 → abs → tampil
```

Filter yang sudah ada adalah **ideal high-pass** bergaya *square mask*: matriks satu (`np.ones`) dengan blok tengah `20×20` di-set 0. Sel ini **dibiarkan utuh sebagai acuan** dan tidak diubah.

## Lingkup

Menambahkan **tiga fungsi filter baru** plus helper dan visualisasi perbandingan. Alur fft/ifft tidak berubah — hanya bentuk filter yang berbeda.

### Konsep kunci
- **Low-pass** = pertahankan frekuensi rendah (tengah) → efek menghaluskan / blur.
- **High-pass** = pertahankan frekuensi tinggi (tepi) → efek menonjolkan tepi / detail.
- Ideal filter (square) = transisi tajam (0/1). Gaussian filter = transisi mulus.

## Komponen

### 1. Helper distance map (untuk Gaussian)
Membuat matriks jarak `D(u,v)` dari titik tengah:

```
D[u,v] = sqrt((u - center_row)^2 + (v - center_col)^2)
```

Diimplementasikan dengan `np.meshgrid` agar vektor (tanpa loop).

### 2. Tiga fungsi filter

| Fungsi | Gaya | Logika |
|---|---|---|
| `ideal_lowpass(shape, size=10)` | square | matriks **nol** (`np.zeros`), blok tengah `2*size × 2*size` di-set **1**. Kebalikan dari ideal high-pass acuan. |
| `gaussian_lowpass(shape, D0=30)` | distance | `H = exp(-D^2 / (2*D0^2))`. Tengah ≈1, makin jauh →0. |
| `gaussian_highpass(shape, D0=30)` | distance | `H = 1 - exp(-D^2 / (2*D0^2))`. Kebalikan Gaussian low-pass. |

Catatan: `ideal_highpass` **tidak** dibuat sebagai fungsi — sel yang sudah ada tetap jadi acuan eksplisit.

### 3. Fungsi apply + visualisasi
- `apply_filter(img_dft, filter)` → kembalikan citra hasil (ifftshift → ifft2 → abs).
- Satu sel visualisasi grid: untuk tiap filter tampilkan **(mask filter, spektrum setelah difilter, citra hasil)**.

## Parameter default
- Ideal: `size = 10` (blok 20×20, sama seperti acuan).
- Gaussian: `D0 = 30` (radius cutoff).
- Mudah diubah untuk eksperimen.

## Kriteria sukses
- Ketiga filter berjalan tanpa error pada `assets/character.tif`.
- Ideal low-pass menghasilkan citra blur; Gaussian low-pass blur lebih mulus (tanpa ringing).
- Gaussian high-pass menghasilkan citra tepi/detail.
- Visualisasi perbandingan tampil dalam satu grid.

## Di luar lingkup
- Tidak mengubah sel ideal high-pass yang sudah ada.
- Tidak menambah filter Butterworth atau jenis lain.
