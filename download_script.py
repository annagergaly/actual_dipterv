import download_abide_preproc

base_path = "C:/Users/Anna/Documents/actual_dipterv/data"
sites = ["PITT", "OLIN", "OHSU", "SDSU", "TRINITY", "UM_2", "YALE", "CMU", "LEUVEN_1", "LEUVEN_2", "KKI", "STANFORD", "UCLA_1", "UCLA_2", "MAX_MUN", "CALTECH", "SBL"]

for site in sites:
    # download_abide_preproc.collect_and_download(
    #     derivative="rois_ho", 
    #     pipeline="cpac", 
    #     strategy="filt_noglobal",
    #     out_dir=f"{base_path}/{site}",
    #     less_than=200.0,
    #     greater_than=-1.0,
    #     site=site,
    #     sex=None,
    #     diagnosis="asd"
    #     )
    download_abide_preproc.collect_and_download(
        derivative="rois_ho", 
        pipeline="cpac", 
        strategy="filt_noglobal",
        out_dir=f"{base_path}/{site}_control",
        less_than=200.0,
        greater_than=-1.0,
        site=site,
        sex=None,
        diagnosis="tdc"
        )
