import re
import sys
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin


# ==============================================================================
# 1. DEFINISI CLASS HARUS ADA DI __main__ AGAR JOBLIB BISA MEMUAT PIPELINE
# ==============================================================================
class SiakadFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def _clean_text(val):
        if pd.isna(val):
            return ""
        return str(val).strip().lower()

    @staticmethod
    def _get_col(df_in, candidates):
        for c in candidates:
            if c in df_in.columns:
                return df_in[c]
        raise KeyError(f"Kolom tidak ditemukan: {candidates}")

    def _map_jurusan_polbeng(self, prodi):
        t = self._clean_text(prodi)
        if "perkapalan" in t: return "Jurusan Teknik Perkapalan"
        elif "mesin" in t: return "Jurusan Teknik Mesin"
        elif "listrik" in t or "elektronika" in t: return "Jurusan Teknik Elektro"
        elif "sipil" in t or "jalan dan jembatan" in t: return "Jurusan Teknik Sipil"
        elif "bisnis" in t or "akuntansi" in t: return "Jurusan Administrasi Niaga"
        elif "informatika" in t or "keamanan sistem" in t or "perangkat lunak" in t: return "Jurusan Teknik Informatika"
        elif "bahasa inggris" in t: return "Jurusan Bahasa"
        elif "pelayaran" in t or "nautika" in t: return "Jurusan Kemaritiman"
        return "Lainnya"

    def _map_jenjang(self, prodi):
        t = self._clean_text(prodi)
        if t.startswith("d4") or "d-4" in t or "sarjana terapan" in t:
            return "D4"
        return "D3"

    def _map_status_pilihan(self, row):
        diterima = self._clean_text(row.get("Prodi Diterima", ""))
        p1 = self._clean_text(row.get("Pilihan 1", ""))
        p2 = self._clean_text(row.get("Pilihan 2", ""))
        if diterima and diterima == p1: return "Pilihan_1"
        elif diterima and diterima == p2: return "Pilihan_2"
        return "Lintas_Pilihan"

    def _map_jenis_sekolah(self, jenis):
        t = self._clean_text(jenis)
        if "smk" in t: return "SMK"
        elif "ma" in t or "madrasah" in t: return "MA"
        elif "sma" in t: return "SMA"
        return "Lainnya"

    def _map_linearitas(self, jurusan_sekolah, jurusan_polbeng):
        js = self._clean_text(jurusan_sekolah)
        jp = jurusan_polbeng
        jurusan_teknik = {
            "Jurusan Teknik Perkapalan", "Jurusan Teknik Mesin",
            "Jurusan Teknik Elektro", "Jurusan Teknik Sipil", "Jurusan Teknik Informatika",
        }
        kata_ipa = [
            "ipa", "mia", "teknik", "komputer", "tkj", "rpl", "multimedia",
            "mesin", "listrik", "elektronika", "otomotif", "bangunan", "sipil",
            "perkapalan", "nautika", "pelayaran", "ti", "it"
        ]
        kata_ips = [
            "ips", "iis", "akuntansi", "pemasaran", "perkantoran", "bisnis",
            "manajemen", "ekonomi", "keuangan", "perbankan", "administrasi"
        ]
        kata_bahasa = ["bahasa", "ibb", "pariwisata", "perhotelan", "sastra"]

        if jp in jurusan_teknik:
            return "Linear" if any(k in js for k in kata_ipa) else "Non_Linear"
        elif jp == "Jurusan Administrasi Niaga":
            return "Linear" if any(k in js for k in kata_ips) else "Non_Linear"
        elif jp == "Jurusan Bahasa":
            return "Linear" if any(k in js for k in kata_bahasa) else "Non_Linear"
        elif jp == "Jurusan Kemaritiman":
            return "Linear" if any(k in js for k in kata_ipa + kata_ips) else "Non_Linear"
        return "Non_Linear"

    def _map_wilayah(self, kota, provinsi):
        k = self._clean_text(kota)
        p = self._clean_text(provinsi)
        if "bengkalis" in k: return "Lokal_Bengkalis"
        elif "riau" in p and "kepulauan riau" not in p: return "Dalam_Riau"
        return "Luar_Riau"

    def _map_pekerjaan(self, kerja, is_ibu=False):
        t = self._clean_text(kerja)
        if not t or t in ["-", "nan"]:
            return "IRT_Tidak_Bekerja" if is_ibu else "Tidak_Bekerja"
        kata_tidak_kerja = [
            "tidak bekerja", "belum bekerja", "ibu rumah tangga", "irt",
            "mengurus rumah tangga", "meninggal", "alm", "almarhum", "pensiun", "pelajar"
        ]
        kata_formal = [
            "pns", "pegawai negeri", "asn", "tni", "polri", "bumn", "bumd",
            "karyawan swasta", "pegawai swasta", "guru", "dosen", "dokter",
            "bidan", "perawat", "tenaga medis", "pegawai"
        ]
        kata_informal = [
            "wiraswasta", "wirausaha", "pedagang", "petani", "pekebun",
            "nelayan", "buruh", "tukang", "sopir", "supir", "ojek",
            "pekerja lepas", "freelance", "serabutan", "honorer", "bekerja"
        ]
        if any(k in t for k in kata_tidak_kerja):
            return "IRT_Tidak_Bekerja" if is_ibu else "Tidak_Bekerja"
        elif any(k in t for k in kata_formal):
            return "Formal_Stabil"
        elif any(k in t for k in kata_informal):
            return "Informal_Fluktuatif"
        return "Lainnya"

    def _parse_penghasilan(self, val):
        if pd.isna(val): return np.nan
        t = self._clean_text(val)
        if any(k in t for k in ["tidak berpenghasilan", "tidak ada", "nol"]) or t == "0":
            return 0.0
        t_clean = re.sub(r"[,\.]00\b", "", t).replace(".", "").replace(",", "")
        angka_list = [float(x) for x in re.findall(r"\d+", t_clean)]
        if not angka_list: return np.nan
        if "juta" in t_clean:
            angka_list = [a * 1_000_000 if a < 1000 else a for a in angka_list]
        elif "ribu" in t_clean:
            angka_list = [a * 1_000 if a < 10000 else a for a in angka_list]
        if len(angka_list) >= 2:
            return (angka_list[0] + angka_list[1]) / 2.0
        val_num = angka_list[0]
        if "kurang" in t_clean or "<" in t_clean: return val_num * 0.5
        elif "lebih" in t_clean or ">" in t_clean: return val_num * 1.2
        return val_num

    def _parse_nominal_ukt(self, val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        s = str(val).strip()
        if "," in s: s = s.split(",")[0]
        s = re.sub(r"\.00$", "", s)
        digits = re.sub(r"[^\d]", "", s)
        return float(digits) if digits else np.nan

    def _extract_tahun_periode(self, val):
        if pd.isna(val): return np.nan
        m = re.search(r"(20\d{2})", str(val))
        return int(m.group(1)) if m else np.nan

    def _extract_tahun_lahir(self, val):
        if pd.isna(val): return np.nan
        dt = pd.to_datetime(val, errors="coerce")
        if pd.notna(dt): return int(dt.year)
        m = re.search(r"(19\d{2}|20\d{2})", str(val))
        return int(m.group(1)) if m else np.nan

    def _map_gelombang(self, val):
        t = self._clean_text(val)
        if re.search(r"\b1\b|gelombang\s*1|gel\.?\s*1", t): return "Gelombang_1"
        elif re.search(r"\b2\b|gelombang\s*2|gel\.?\s*2", t): return "Gelombang_2"
        return "Gelombang_Lainnya"

    def transform(self, X):
        df_in = X.copy()
        df_in.columns = [str(c).strip() for c in df_in.columns]
        out = pd.DataFrame(index=df_in.index)

        prodi_diterima = self._get_col(df_in, ["Prodi Diterima"])
        out["Jurusan_Polbeng"] = prodi_diterima.apply(self._map_jurusan_polbeng)
        out["Jenjang_Studi"] = prodi_diterima.apply(self._map_jenjang)
        out["Status_Pilihan"] = df_in.apply(self._map_status_pilihan, axis=1)
        out["Kategori_Sekolah"] = self._get_col(df_in, ["Jenis Institusi"]).apply(self._map_jenis_sekolah)
        out["Linearitas_Jurusan"] = [
            self._map_linearitas(js, jp)
            for js, jp in zip(self._get_col(df_in, ["Jurusan"]), out["Jurusan_Polbeng"])
        ]

        jk = self._get_col(df_in, ["Jenis Kelamin", "L/P"]).apply(self._clean_text)
        out["Jenis_Kelamin_Numerik"] = np.where(jk.str.startswith("l"), 1, 0)
        out["Kategori_Wilayah_Asal"] = [
            self._map_wilayah(k, p)
            for k, p in zip(self._get_col(df_in, ["Kota"]), self._get_col(df_in, ["Provinsi"]))
        ]

        tahun_masuk = self._get_col(df_in, ["Periode"]).apply(self._extract_tahun_periode)
        tahun_lahir = self._get_col(df_in, ["Tanggal Lahir"]).apply(self._extract_tahun_lahir)
        usia = tahun_masuk - tahun_lahir
        out["Usia_Saat_Mendaftar"] = usia.where((usia >= 15) & (usia <= 45), np.nan)

        tahun_lulus = pd.to_numeric(self._get_col(df_in, ["Tahun Lulus"]), errors="coerce")
        gap = tahun_masuk - tahun_lulus
        out["Lama_Gap_Year"] = gap.where((gap >= 0) & (gap <= 15), np.nan)

        out["Kategori_Pekerjaan_Ayah"] = self._get_col(df_in, ["Pekerjaan Ayah"]).apply(
            lambda x: self._map_pekerjaan(x, is_ibu=False)
        )
        out["Kategori_Pekerjaan_Ibu"] = self._get_col(df_in, ["Pekerjaan Ibu"]).apply(
            lambda x: self._map_pekerjaan(x, is_ibu=True)
        )

        ayah_kerja = (out["Kategori_Pekerjaan_Ayah"] != "Tidak_Bekerja").astype(int)
        ibu_kerja = (out["Kategori_Pekerjaan_Ibu"] != "IRT_Tidak_Bekerja").astype(int)
        out["Jumlah_Ortu_Bekerja"] = ayah_kerja + ibu_kerja

        gaji_ayah = self._get_col(df_in, ["Penghasilan Ayah"]).apply(self._parse_penghasilan).fillna(0.0)
        gaji_ibu = self._get_col(df_in, ["Penghasilan Ibu"]).apply(self._parse_penghasilan).fillna(0.0)
        out["Total_Penghasilan_Ortu"] = gaji_ayah + gaji_ibu

        out["Nominal_UKT"] = self._get_col(df_in, ["Nominal UKT"]).apply(self._parse_nominal_ukt)

        kip = self._get_col(df_in, ["Pengusul KIP", "KIP Kuliah"]).apply(self._clean_text)
        out["Status_Pengusul_KIP"] = np.where(
            kip.isin(["ya", "yes", "1", "true", "pengusul", "iya"]), 1, 0
        )

        pembagi_aman = np.where(
            out["Total_Penghasilan_Ortu"] > 0, out["Total_Penghasilan_Ortu"], 100_000.0
        )
        out["Rasio_Beban_UKT"] = out["Nominal_UKT"] / pembagi_aman
        out["Gelombang_Kategori"] = self._get_col(df_in, ["Gelombang"]).apply(self._map_gelombang)
        return out


# Daftarkan class ke __main__ agar uvicorn bisa memuat pickle dari Colab
sys.modules["__main__"].SiakadFeatureEngineer = SiakadFeatureEngineer

# Muat paket model final saat server menyala
paket = joblib.load("model_komitmen_polbeng_2027.joblib")
pipeline = paket["pipeline"]
TH_RENDAH = paket["threshold_rendah"]
TH_SEDANG = paket["threshold_sedang"]
REKOM_MAP = paket["rekomendasi_map"]

app = FastAPI(title="API Prediksi Komitmen Mahasiswa Polbeng")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatchRequest(BaseModel):
    records: List[Dict[str, Any]]


@app.get("/")
def health_check():
    return {"status": "online", "model": "Polbeng Komitmen 2027 (RF Balanced + Threshold)"}


@app.post("/predict")
def predict_batch(payload: BatchRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="Data records kosong.")

    try:
        df_input = pd.DataFrame(payload.records)
        probs = pipeline.predict_proba(df_input)

        hasil = []
        for p0, p1, p2 in probs:
            if p1 >= TH_SEDANG:
                label = "Sedang"
            elif p0 >= TH_RENDAH:
                label = "Rendah"
            else:
                label = "Tinggi"

            hasil.append({
                "Prediksi_Komitmen": label,
                "Prob_Rendah": round(float(p0) * 100, 1),
                "Prob_Sedang": round(float(p1) * 100, 1),
                "Prob_Tinggi": round(float(p2) * 100, 1),
                "Rekomendasi": REKOM_MAP[label],
            })

        return {"status": "success", "total": len(hasil), "predictions": hasil}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))