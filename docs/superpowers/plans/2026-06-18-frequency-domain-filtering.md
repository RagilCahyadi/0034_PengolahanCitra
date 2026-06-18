# Frequency Domain Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Menambahkan tiga filter frequency domain baru (ideal low-pass square, Gaussian low-pass, Gaussian high-pass) ke `w10_FrequencyDomain.ipynb`, mengikuti logika ideal high-pass yang sudah ada.

**Architecture:** Alur fft/ifft yang sudah ada (`fft2 → fftshift → ×filter → ifftshift → ifft2 → abs`) dipertahankan; hanya bentuk filter yang baru. Filter dibungkus sebagai fungsi reusable. Ideal filter pakai gaya *square mask*; Gaussian pakai *distance map* `D(u,v)`.

**Tech Stack:** Python, NumPy (`np.fft`, `np.meshgrid`), OpenCV (`cv2`), Matplotlib.

## Global Constraints

- File target tunggal: `w10_FrequencyDomain.ipynb`. Cell ideal high-pass yang sudah ada (id `42cd7e63`) **TIDAK** diubah — dibiarkan sebagai acuan.
- Tidak membuat fungsi `ideal_highpass`.
- Gaya verifikasi: **assertion di dalam cell notebook** (bukan pytest). "Run test" = jalankan cell; gagal = NameError/AssertionError, lulus = cell jalan tanpa error.
- Parameter default: ideal `size=10`, Gaussian `D0=30`.
- Commit git **opsional** (user tidak mewajibkan push). Akhiri tiap task dengan menyimpan notebook; commit hanya bila user minta.
- Citra uji: `./assets/character.tif` (sudah dipakai cell pertama, variabel `img` & `img_dft` tersedia).

---

### Task 1: Helper distance map

**Files:**
- Modify: `w10_FrequencyDomain.ipynb` (tambah cell baru setelah cell terakhir)

**Interfaces:**
- Consumes: `numpy as np` (sudah di-import di cell pertama).
- Produces: `distance_map(shape) -> np.ndarray` — mengembalikan matriks float64 berukuran `shape` berisi jarak Euclidean tiap piksel ke titik tengah.

- [ ] **Step 1: Tulis cell pengecekan (gagal dulu)**

Tambah cell baru berisi assertion yang memakai `distance_map` (belum ada):

```python
# CEK Task 1 (jalankan dulu sebelum implementasi -> harus error NameError)
_d = distance_map((5, 5))
assert _d.shape == (5, 5)
assert _d[2, 2] == 0.0            # titik tengah jaraknya 0
assert round(_d[2, 0], 3) == 2.0  # 2 piksel ke kiri dari tengah
print("Task 1 OK")
```

- [ ] **Step 2: Jalankan cell cek → pastikan gagal**

Expected: `NameError: name 'distance_map' is not defined`.

- [ ] **Step 3: Tulis implementasi minimal**

Tambah cell baru SEBELUM cell cek, berisi:

```python
def distance_map(shape):
    rows, columns = shape
    center_row = rows // 2
    center_column = columns // 2
    # u = indeks baris, v = indeks kolom
    u = np.arange(rows).reshape(-1, 1)   # kolom vektor
    v = np.arange(columns).reshape(1, -1)  # baris vektor
    d = np.sqrt((u - center_row) ** 2 + (v - center_column) ** 2)
    return d
```

- [ ] **Step 4: Jalankan ulang cell cek → pastikan lulus**

Expected: mencetak `Task 1 OK` tanpa error.

- [ ] **Step 5: Simpan notebook** (commit opsional, hanya bila user minta)

---

### Task 2: Ideal low-pass (square)

**Files:**
- Modify: `w10_FrequencyDomain.ipynb` (tambah cell baru)

**Interfaces:**
- Consumes: `numpy as np`.
- Produces: `ideal_lowpass(shape, size=10) -> np.ndarray` — matriks uint8 ukuran `shape`, nol di mana-mana kecuali blok tengah `2*size × 2*size` bernilai 1. (Kebalikan dari ideal high-pass acuan.)

- [ ] **Step 1: Tulis cell pengecekan (gagal dulu)**

```python
# CEK Task 2
_f = ideal_lowpass((100, 100), size=10)
assert _f.shape == (100, 100)
assert _f[50, 50] == 1            # tengah = lolos (low freq)
assert _f[0, 0] == 0             # pojok = blok (high freq)
assert _f.sum() == 20 * 20       # luas blok tengah 20x20
print("Task 2 OK")
```

- [ ] **Step 2: Jalankan → pastikan gagal**

Expected: `NameError: name 'ideal_lowpass' is not defined`.

- [ ] **Step 3: Tulis implementasi minimal** (cell baru sebelum cell cek)

```python
def ideal_lowpass(shape, size=10):
    rows, columns = shape
    center_row = rows // 2
    center_column = columns // 2
    f = np.zeros((rows, columns), np.uint8)
    f[center_row - size:center_row + size,
      center_column - size:center_column + size] = 1
    return f
```

- [ ] **Step 4: Jalankan ulang cell cek → pastikan lulus**

Expected: mencetak `Task 2 OK`.

- [ ] **Step 5: Simpan notebook**

---

### Task 3: Gaussian low-pass & high-pass

**Files:**
- Modify: `w10_FrequencyDomain.ipynb` (tambah cell baru)

**Interfaces:**
- Consumes: `distance_map` (Task 1), `numpy as np`.
- Produces:
  - `gaussian_lowpass(shape, D0=30) -> np.ndarray` float64: `exp(-D^2 / (2*D0^2))`.
  - `gaussian_highpass(shape, D0=30) -> np.ndarray` float64: `1 - gaussian_lowpass`.

- [ ] **Step 1: Tulis cell pengecekan (gagal dulu)**

```python
# CEK Task 3
_glp = gaussian_lowpass((100, 100), D0=30)
_ghp = gaussian_highpass((100, 100), D0=30)
assert _glp.shape == (100, 100)
assert round(_glp[50, 50], 6) == 1.0           # tengah = 1 (D=0)
assert _glp[0, 0] < _glp[50, 50]               # makin jauh makin kecil
assert np.allclose(_glp + _ghp, 1.0)           # low + high = 1
assert round(_ghp[50, 50], 6) == 0.0           # tengah high-pass = 0
print("Task 3 OK")
```

- [ ] **Step 2: Jalankan → pastikan gagal**

Expected: `NameError: name 'gaussian_lowpass' is not defined`.

- [ ] **Step 3: Tulis implementasi minimal** (cell baru sebelum cell cek)

```python
def gaussian_lowpass(shape, D0=30):
    d = distance_map(shape)
    return np.exp(-(d ** 2) / (2 * (D0 ** 2)))

def gaussian_highpass(shape, D0=30):
    return 1 - gaussian_lowpass(shape, D0)
```

- [ ] **Step 4: Jalankan ulang cell cek → pastikan lulus**

Expected: mencetak `Task 3 OK`.

- [ ] **Step 5: Simpan notebook**

---

### Task 4: apply_filter + visualisasi perbandingan

**Files:**
- Modify: `w10_FrequencyDomain.ipynb` (tambah cell baru)

**Interfaces:**
- Consumes: `img`, `img_dft` (cell pertama), `ideal_lowpass`, `gaussian_lowpass`, `gaussian_highpass`, `matplotlib.pyplot as plt`, `numpy as np`.
- Produces: `apply_filter(img_dft, filter) -> np.ndarray` — citra hasil (magnitude) setelah inverse transform. Plus satu cell visualisasi grid.

- [ ] **Step 1: Tulis cell pengecekan apply_filter (gagal dulu)**

```python
# CEK Task 4 (apply_filter)
_res = apply_filter(img_dft, ideal_lowpass(img.shape, size=10))
assert _res.shape == img.shape
assert np.isrealobj(_res)      # hasil sudah magnitude (real)
print("Task 4a OK")
```

- [ ] **Step 2: Jalankan → pastikan gagal**

Expected: `NameError: name 'apply_filter' is not defined`.

- [ ] **Step 3: Tulis implementasi minimal** (cell baru sebelum cell cek)

```python
def apply_filter(img_dft, filter):
    filtered = img_dft * filter
    filtered = np.fft.ifftshift(filtered)
    filtered = np.fft.ifft2(filtered)
    return np.abs(filtered)
```

- [ ] **Step 4: Jalankan ulang cell cek → pastikan lulus**

Expected: mencetak `Task 4a OK`.

- [ ] **Step 5: Tulis cell visualisasi grid**

Tambah cell baru:

```python
filters = {
    'Ideal Low-Pass': ideal_lowpass(img.shape, size=10),
    'Gaussian Low-Pass': gaussian_lowpass(img.shape, D0=30),
    'Gaussian High-Pass': gaussian_highpass(img.shape, D0=30),
}

fig, axes = plt.subplots(len(filters), 3, figsize=(12, 4 * len(filters)))

for row, (name, f) in enumerate(filters.items()):
    result = apply_filter(img_dft, f)
    spectrum = np.log(np.abs(img_dft * f) + 1)  # +1 hindari log(0)

    axes[row, 0].imshow(f, cmap='gray')
    axes[row, 0].set_title(f'{name}\n(Filter Mask)')

    axes[row, 1].imshow(spectrum, cmap='gray')
    axes[row, 1].set_title('Spektrum setelah filter')

    axes[row, 2].imshow(result, cmap='gray')
    axes[row, 2].set_title('Citra hasil')

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()
```

- [ ] **Step 6: Jalankan cell visualisasi → verifikasi mata**

Expected: grid 3×3 tampil. Ideal low-pass → citra blur (mungkin ada ringing). Gaussian low-pass → blur lebih mulus. Gaussian high-pass → tepi/detail menonjol.

- [ ] **Step 7: Simpan notebook**

---

## Catatan urutan cell di notebook

Agar fungsi terdefinisi sebelum dipakai, urutan akhir cell (setelah cell acuan yang sudah ada):
1. `distance_map` (Task 1 impl)
2. `ideal_lowpass` (Task 2 impl)
3. `gaussian_lowpass` + `gaussian_highpass` (Task 3 impl)
4. `apply_filter` (Task 4 impl)
5. cell visualisasi grid (Task 4 step 5)

Cell-cell "CEK" boleh dihapus setelah lulus, atau dibiarkan sebagai dokumentasi. Selama eksekusi, jalankan cell sesuai urutan (Kernel → Restart & Run All untuk verifikasi akhir).
