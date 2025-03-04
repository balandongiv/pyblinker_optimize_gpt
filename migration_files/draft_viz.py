def viz_fit_blink_pd_approach(data=None,df=None):
    """

    TODO Viz

    https://stackoverflow.com/a/51928241/6446053
    https://stackoverflow.com/a/38015084/6446053
    :return:
    """
    from pyblinkers.viz_pd import viz_complete_blink_prop
    import mne



    title = 'sddd'
    rep = mne.Report(title=title)
    print(df.dtypes)
    cols_int = ['rightBase']
    df[cols_int] = df[cols_int].astype(int)
    df.to_excel('da.xlsx')
    fig_good_blink = []
    fig_bad_blink = []
    # for index, row in df.iloc[1:].iterrows():
    for index, row in df.iterrows():

        dfig = viz_complete_blink_prop(data, row)

        if row['blink_quality'] == 'Good':
            fig_good_blink.append(dfig)
        else:
            fig_bad_blink.append(dfig)

    all_cap_good = ['Good'] * len(fig_good_blink)
    for disfig, discaption in zip(fig_good_blink, all_cap_good):
        # lcaption=discaption
        rep.add_figs_to_section(disfig, captions=discaption, section='Good blink')

    all_cap_bad = ['Good'] * len(fig_bad_blink)
    for disfig, discaption in zip(fig_bad_blink, all_cap_bad):
        # lcaption=discaption
        rep.add_figs_to_section(disfig, captions=discaption, section='bad blink')

    spath = 'dreport_testdddd.html'
    rep.save(spath, overwrite=True, open_browser=False)


import hickle as hkl

filename = '_draft_data_to_viz_complete.hkl'
raw, params, ch_selected, all_data_info= hkl.load(filename)

ch=ch_selected.loc[0,'ch']
data=raw.get_data(picks=ch)[0]
k=list(filter(lambda all_data_info: all_data_info['ch'] == ch, all_data_info))[0]
df=k['df']
viz_fit_blink_pd_approach(data=data,df=df)
# hkl.dump([raw, params, ch_selected, all_data_info], '_draft_data_to_viz_complete.hkl')