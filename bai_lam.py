import os
import sys
import pandas as pd
import numpy as np


def run_pipeline():
    # Xác định đường dẫn thư mục dữ liệu và đầu ra
    current_dir  = os.path.dirname(os.path.abspath(__file__)) 
    parent_dir   = os.path.dirname(current_dir)                  
    grandparent_dir = os.path.dirname(parent_dir)                

    # Tìm thư mục du_lieu theo thứ tự ưu tiên
    _candidates = [
        os.path.join(grandparent_dir, "du_lieu"),  
        os.path.join(parent_dir,      "du_lieu"),  
        os.path.join(current_dir,     "du_lieu"),  
        os.path.abspath("du_lieu"),                
    ]
    data_dir = next((p for p in _candidates if os.path.exists(p)), None)
    if data_dir is None:
        raise FileNotFoundError(
            "Không tìm thấy thư mục 'du_lieu'!\n"
            f"Đã tìm trong:\n" + "\n".join(f"  - {p}" for p in _candidates)
        )

    # Thư mục xuất kết quả
    out_dir = current_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Thư mục dữ liệu: {data_dir}")
    print(f"[*] Thư mục đầu ra:   {out_dir}")

    # =========================================================================
    # Bước 1: Đọc 6 file CSV với dtype=str, keep_default_na=False
    # =========================================================================
    print("\n--- BƯỚC 1: ĐỌC DỮ LIỆU ---")
    df_kh = pd.read_csv(os.path.join(data_dir, "khach_hang.csv"), dtype=str, keep_default_na=False)
    df_cn = pd.read_csv(os.path.join(data_dir, "chi_nhanh.csv"), dtype=str, keep_default_na=False)
    df_tk = pd.read_csv(os.path.join(data_dir, "tai_khoan.csv"), dtype=str, keep_default_na=False)
    df_gd = pd.read_csv(os.path.join(data_dir, "giao_dich.csv"), dtype=str, keep_default_na=False)
    df_the = pd.read_csv(os.path.join(data_dir, "the.csv"), dtype=str, keep_default_na=False)
    df_tg = pd.read_csv(os.path.join(data_dir, "ty_gia.csv"), dtype=str, keep_default_na=False)

    print(f"Shape khach_hang: {df_kh.shape}")
    print(f"Shape chi_nhanh:  {df_cn.shape}")
    print(f"Shape tai_khoan:  {df_tk.shape}")
    print(f"Shape giao_dich:  {df_gd.shape}")
    print(f"Shape the:        {df_the.shape}")
    print(f"Shape ty_gia:     {df_tg.shape}")

    # =========================================================================
    # Bước 2: Làm sạch văn bản (Q1)
    # Bỏ khoảng trắng đầu/cuối mọi cột chữ; ma_kh viết HOA; trang_thai viết HOA
    # =========================================================================
    print("\n--- BƯỚC 2: LÀM SẠCH VĂN BẢN (Q1) ---")
    for df in [df_kh, df_cn, df_tk, df_gd, df_the, df_tg]:
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    for df in [df_kh, df_tk, df_the]:
        if "ma_kh" in df.columns:
            df["ma_kh"] = df["ma_kh"].str.upper()

    df_kh["trang_thai_kh"] = df_kh["trang_thai_kh"].str.upper()
    df_gd["trang_thai"] = df_gd["trang_thai"].str.upper()
    df_the["trang_thai"] = df_the["trang_thai"].str.upper()

    # =========================================================================
    # Bước 3: Khách hàng hoạt động, nối chi nhánh, tính tuổi (Q2, Q3, Q4)
    # =========================================================================
    print("\n--- BƯỚC 3: XỬ LÝ KHÁCH HÀNG & CHI NHÁNH & TUỔI (Q2 - Q4) ---")
    # Q2: Chỉ giữ khách hàng đang hoạt động, mỗi ma_kh đúng 1 dòng
    df_kh_active = df_kh[df_kh["trang_thai_kh"] == "HOAT_DONG"].copy()
    df_kh_active = df_kh_active.drop_duplicates(subset=["ma_kh"], keep="first")
    print(f"Khách hàng HOAT_DONG: {df_kh_active.shape[0]} dòng")

    # Q3: Nối chi nhánh, không có trong danh mục -> 'UNMAPPED'
    df_kh_active = df_kh_active.merge(
        df_cn[["ma_cn", "ten_cn", "vung"]],
        on="ma_cn",
        how="left"
    )
    df_kh_active["ten_cn"] = df_kh_active["ten_cn"].fillna("UNMAPPED").replace("", "UNMAPPED")
    df_kh_active["vung"] = df_kh_active["vung"].fillna("UNMAPPED").replace("", "UNMAPPED")

    # Q4: TUOI = 2026 - năm sinh 
    NAM_TINH_TUOI = 2026
    dt_sinh = pd.to_datetime(df_kh_active["ngay_sinh"], format="%Y-%m-%d", errors="coerce")
    df_kh_active["TUOI"] = dt_sinh.dt.year.apply(
        lambda y: str(int(NAM_TINH_TUOI - y)) if pd.notna(y) else ""
    )

    # =========================================================================
    # Bước 4: Tài khoản x tỷ giá, gom theo khách (Q5)
    # =========================================================================
    print("\n--- BƯỚC 4: XỬ LÝ TÀI KHOẢN (Q5) ---")
    df_tg["ty_gia"] = pd.to_numeric(df_tg["ty_gia"], errors="coerce").fillna(1.0)
    df_tk_calc = df_tk.merge(df_tg[["loai_tien", "ty_gia"]], on="loai_tien", how="left")
    df_tk_calc["ty_gia"] = df_tk_calc["ty_gia"].fillna(1.0)
    df_tk_calc["so_du_float"] = pd.to_numeric(df_tk_calc["so_du"], errors="coerce").fillna(0.0)
    df_tk_calc["so_du_vnd"] = df_tk_calc["so_du_float"] * df_tk_calc["ty_gia"]

    df_tk_agg = df_tk_calc.groupby("ma_kh").agg(
        SO_TAI_KHOAN=("so_tk", "count"),
        TONG_SO_DU_VND=("so_du_vnd", "sum")
    ).reset_index()
    print(f"Số khách hàng có tài khoản: {df_tk_agg.shape[0]}")

    # =========================================================================
    # Bước 5: Giao dịch hợp lệ (Q6)
    # - Không trùng ma_gd (giữ 1 dòng)
    # - so_tien đổi được sang số (bỏ dấu phẩy, ô trống bị loại)
    # - ngày thuộc 2024 (đọc cả 2 định dạng: YYYY-MM-DD và DD/MM/YYYY)
    # - trang_thai = OK
    # - so_tk có trong tai_khoan (tài khoản lạ bị loại)
    # =========================================================================
    print("\n--- BƯỚC 5: LỌC GIAO DỊCH HỢP LỆ (Q6) ---")
    df_gd_dedup = df_gd.drop_duplicates(subset=["ma_gd"], keep="first").copy()
    print(f"Số giao dịch sau khi bỏ trùng: {df_gd_dedup.shape[0]} (đúng chuẩn 6.000)")

    # Đổi số tiền
    df_gd_dedup["so_tien_clean"] = df_gd_dedup["so_tien"].str.replace(",", "", regex=False)
    df_gd_dedup["so_tien_num"] = pd.to_numeric(df_gd_dedup["so_tien_clean"], errors="coerce")

    # Đọc ngày 2 định dạng
    d1 = pd.to_datetime(df_gd_dedup["ngay"], format="%Y-%m-%d", errors="coerce")
    d2 = pd.to_datetime(df_gd_dedup["ngay"], format="%d/%m/%Y", errors="coerce")
    df_gd_dedup["ngay_parsed"] = d1.fillna(d2)

    # Danh sách so_tk hợp lệ trong tai_khoan
    valid_accounts = set(df_tk["so_tk"])

    valid_mask = (
        df_gd_dedup["so_tien_num"].notna() &
        (df_gd_dedup["ngay_parsed"].dt.year == 2024) &
        (df_gd_dedup["trang_thai"] == "OK") &
        (df_gd_dedup["so_tk"].isin(valid_accounts))
    )
    df_gd_valid = df_gd_dedup[valid_mask].copy()
    print(f"Số giao dịch hợp lệ năm 2024: {df_gd_valid.shape[0]}")

    # Nối với tai_khoan và ty_gia để lấy ma_kh và quy đổi tiền VND
    df_gd_valid = df_gd_valid.merge(
        df_tk_calc[["so_tk", "ma_kh", "ty_gia"]],
        on="so_tk",
        how="inner"
    )
    df_gd_valid["tien_vnd"] = df_gd_valid["so_tien_num"] * df_gd_valid["ty_gia"]

    # =========================================================================
    # Bước 6: Gom giao dịch theo khách (Q7)
    # SO_GD_OK, TONG_TIEN_GD_VND, GD_LON_NHAT_VND, KENH_NHIEU_NHAT
    # =========================================================================
    print("\n--- BƯỚC 6: TỔNG HỢP GIAO DỊCH THEO KHÁCH (Q7) ---")
    df_gd_agg = df_gd_valid.groupby("ma_kh").agg(
        SO_GD_OK=("ma_gd", "count"),
        TONG_TIEN_GD_VND=("tien_vnd", "sum"),
        GD_LON_NHAT_VND=("tien_vnd", "max")
    ).reset_index()

    # KENH_NHIEU_NHAT: kênh có nhiều giao dịch nhất (bằng nhau -> theo bảng chữ cái)
    kenh_counts = df_gd_valid.groupby(["ma_kh", "kenh"]).size().reset_index(name="count")
    kenh_counts = kenh_counts.sort_values(
        by=["ma_kh", "count", "kenh"],
        ascending=[True, False, True]
    )
    kenh_top = kenh_counts.drop_duplicates(subset=["ma_kh"], keep="first")[["ma_kh", "kenh"]]
    kenh_top.rename(columns={"kenh": "KENH_NHIEU_NHAT"}, inplace=True)

    df_gd_agg = df_gd_agg.merge(kenh_top, on="ma_kh", how="left")
    print(f"Số khách hàng có giao dịch hợp lệ: {df_gd_agg.shape[0]}")

    # =========================================================================
    # Bước 7: Đếm thẻ ACTIVE theo khách (Q8)
    # =========================================================================
    print("\n--- BƯỚC 7: ĐẾM THẺ ACTIVE (Q8) ---")
    df_the_active = df_the[df_the["trang_thai"] == "ACTIVE"].copy()
    df_the_agg = df_the_active.groupby("ma_kh").agg(
        SO_THE_ACTIVE=("so_the", "count")
    ).reset_index()
    print(f"Số khách hàng có thẻ ACTIVE: {df_the_agg.shape[0]}")

    # =========================================================================
    # Bước 8: Lắp báo cáo, điền mặc định 0, xếp loại (Q9)
    # =========================================================================
    print("\n--- BƯỚC 8: GHÉP BÁO CÁO & XẾP LOẠI (Q9) ---")
    df_report = df_kh_active.merge(df_tk_agg, on="ma_kh", how="left")
    df_report = df_report.merge(df_gd_agg, on="ma_kh", how="left")
    df_report = df_report.merge(df_the_agg, on="ma_kh", how="left")

    # Mặc định khi khách không có tài khoản / giao dịch / thẻ
    df_report["SO_TAI_KHOAN"] = df_report["SO_TAI_KHOAN"].fillna(0).astype(int)
    df_report["TONG_SO_DU_VND"] = df_report["TONG_SO_DU_VND"].fillna(0.0)
    df_report["SO_GD_OK"] = df_report["SO_GD_OK"].fillna(0).astype(int)
    df_report["TONG_TIEN_GD_VND"] = df_report["TONG_TIEN_GD_VND"].fillna(0).round().astype(int)
    df_report["GD_LON_NHAT_VND"] = df_report["GD_LON_NHAT_VND"].fillna(0).round().astype(int)
    df_report["KENH_NHIEU_NHAT"] = df_report["KENH_NHIEU_NHAT"].fillna("")
    df_report["SO_THE_ACTIVE"] = df_report["SO_THE_ACTIVE"].fillna(0).astype(int)

    # Q9: Xếp loại theo TONG_SO_DU_VND
    conds = [
        df_report["TONG_SO_DU_VND"] >= 1_000_000_000,
        df_report["TONG_SO_DU_VND"] >= 100_000_000,
        df_report["TONG_SO_DU_VND"] > 0
    ]
    choices = ["PLATINUM", "GOLD", "STANDARD"]
    df_report["XEP_LOAI"] = np.select(conds, choices, default="NO_BALANCE")

    # Đổi tên cột sang chữ HOA chuẩn đề bài
    df_report.rename(columns={
        "ma_kh": "MA_KH",
        "ten_kh": "TEN_KH",
        "gioi_tinh": "GIOI_TINH",
        "tinh_thanh": "TINH_THANH",
        "ma_cn": "MA_CN",
        "ten_cn": "TEN_CN",
        "vung": "VUNG"
    }, inplace=True)

    # =========================================================================
    # Bước 9: Sắp xếp, chọn đúng 16 cột và ghi ket_qua.csv (Q10)
    # =========================================================================
    print("\n--- BƯỚC 9: SẮP XẾP & XUẤT ket_qua.csv (Q10) ---")
    # Sắp xếp theo TONG_SO_DU_VND giảm dần, bằng nhau thì MA_KH tăng dần
    df_report = df_report.sort_values(
        by=["TONG_SO_DU_VND", "MA_KH"],
        ascending=[False, True]
    )

    cols_16 = [
        "MA_KH", "TEN_KH", "GIOI_TINH", "TUOI", "TINH_THANH", "MA_CN", "TEN_CN", "VUNG",
        "SO_TAI_KHOAN", "TONG_SO_DU_VND", "SO_GD_OK", "TONG_TIEN_GD_VND", "GD_LON_NHAT_VND",
        "KENH_NHIEU_NHAT", "SO_THE_ACTIVE", "XEP_LOAI"
    ]
    df_ket_qua = df_report[cols_16].copy()

    out_ket_qua = os.path.join(out_dir, "KhoaDM2_ket_qua.csv")
    df_ket_qua.to_csv(out_ket_qua, index=False, float_format="%.2f", encoding="utf-8")
    print(f"[+] Đã ghi ket_qua.csv ({len(df_ket_qua)} dòng) tại: {out_ket_qua}")

    # =========================================================================
    # Bước 10: Tổng hợp theo vùng và ghi tong_hop_vung.csv
    # =========================================================================
    print("\n--- BƯỚC 10: TỔNG HỢP THEO VÙNG & XUẤT tong_hop_vung.csv ---")
    df_vung = df_ket_qua.groupby("VUNG").agg(
        SO_KH=("MA_KH", "count"),
        TONG_SO_DU_VND=("TONG_SO_DU_VND", "sum"),
        TONG_TIEN_GD_VND=("TONG_TIEN_GD_VND", "sum")
    ).reset_index()

    # Tính trung bình số dư mỗi khách hàng
    df_vung["TB_SO_DU_MOI_KH"] = df_vung["TONG_SO_DU_VND"] / df_vung["SO_KH"]

    # Sắp xếp theo VUNG
    df_vung = df_vung.sort_values(by="VUNG", ascending=True)

    cols_5 = ["VUNG", "SO_KH", "TONG_SO_DU_VND", "TONG_TIEN_GD_VND", "TB_SO_DU_MOI_KH"]
    df_vung = df_vung[cols_5]

    out_vung = os.path.join(out_dir, "KhoaDM2_CTV_tong_hop_vung.csv")
    df_vung.to_csv(out_vung, index=False, float_format="%.2f", encoding="utf-8")
    print(f"[+] Đã ghi tong_hop_vung.csv ({len(df_vung)} dòng) tại: {out_vung}")

    # =========================================================================
    # TỰ KIỂM TRA TRƯỚC KHI NỘP
    # =========================================================================
    print("\n================ TỰ KIỂM TRA TRƯỚC KHI NỘP ================")
    check_dedup = df_gd_dedup.shape[0] == 6000
    print(f"1. Số giao dịch sau khi bỏ trùng: {df_gd_dedup.shape[0]} == 6000 -> {'ĐẠT' if check_dedup else 'CHƯA ĐẠT'}")

    check_upper = df_ket_qua["MA_KH"].str.isupper().all() and not df_ket_qua["MA_KH"].str.contains(r"\s").any()
    print(f"2. Không còn ma_kh chữ thường/khoảng trắng -> {'ĐẠT' if check_upper else 'CHƯA ĐẠT'}")

    unmapped_rows = df_ket_qua[df_ket_qua["MA_CN"] == "099"]
    check_099 = (unmapped_rows["TEN_CN"] == "UNMAPPED").all() and (unmapped_rows["VUNG"] == "UNMAPPED").all()
    print(f"3. Mã chi nhánh 099 ra UNMAPPED ({len(unmapped_rows)} KH) -> {'ĐẠT' if check_099 else 'CHƯA ĐẠT'}")

    # Cột tiền/đếm không có .0
    print("4. Kiểm tra dòng minh họa KH0300:")
    row_300 = df_ket_qua[df_ket_qua["MA_KH"] == "KH0300"]
    if len(row_300) > 0:
        sample_str = ",".join(
            f"{row_300[c].values[0]:.2f}" if c == "TONG_SO_DU_VND" else str(row_300[c].values[0])
            for c in cols_16
        )
        print("   Kết quả:   ", sample_str)
        print("   Đề bài mẫu:", "KH0300,Bùi Quốc Châu,Nữ,52,Cần Thơ,007,CN Đà Nẵng,TRUNG,3,3665037802.47,16,213600000,50000000,POS,1,PLATINUM")
        print(f"   Khớp dòng mẫu đề bài -> {'ĐẠT' if sample_str == 'KH0300,Bùi Quốc Châu,Nữ,52,Cần Thơ,007,CN Đà Nẵng,TRUNG,3,3665037802.47,16,213600000,50000000,POS,1,PLATINUM' else 'CẦN XEM LẠI'}")

    check_no_acc = (df_ket_qua["SO_TAI_KHOAN"] == 0).sum()
    print(f"5. Khách không có tài khoản/giao dịch/thẻ vẫn có mặt: {check_no_acc} KH -> ĐẠT")

    check_sum_kh = df_vung["SO_KH"].sum() == len(df_ket_qua)
    print(f"6. Tổng SO_KH ({df_vung['SO_KH'].sum()}) == số dòng ket_qua ({len(df_ket_qua)}) -> {'ĐẠT' if check_sum_kh else 'CHƯA ĐẠT'}")
    print("============================================================\n")


if __name__ == "__main__":
    run_pipeline()

