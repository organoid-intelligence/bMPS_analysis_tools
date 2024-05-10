from IPython.core.debugger import set_trace

def analyse(df_test):
    session_0_flag = []
    status_aligned = []
    aligned_flag = []
    for i in range(len(df_test)):
        if ((df_test['session'][i]=='0') &( df_test['status_configFile'][i]!='rest')):
            session_0_flag.append('Not started with rest')
            status_aligned.append(1)
            aligned_flag.append(0)
        elif ((df_test['session'][i]=='0') & (df_test['status_configFile'][i]=='rest')):
            session_0_flag.append('Started with rest')
            status_aligned.append(-1)
            aligned_flag.append(0)
        elif (df_test['session'][i]!='0'):
            x = df_test.loc[(df_test['session']=='0') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
            if len(x)>0:
                x = x.item()
                if ( (x == -1) & (int(df_test['session'][i])%2 ==1 ) ):
                    status_aligned.append(-x)
                    aligned_flag.append(0)
                    session_0_flag.append('Started with rest')
                elif ( (x == -1) & (int(df_test['session'][i])%2 ==0 ) ):
                    status_aligned.append(x)
                    aligned_flag.append(0)
                    session_0_flag.append('Started with rest')
                elif ( (x == 1) & (int(df_test['session'][i])%2 ==0 ) ):
                    status_aligned.append(x)
                    aligned_flag.append(0)
                    session_0_flag.append('Not started with rest')
                elif ( (x == 1) & (int(df_test['session'][i])%2 ==1 ) ):
                    status_aligned.append(-x)
                    aligned_flag.append(0)
                    session_0_flag.append('Not started with rest')
            elif (len(x)==0): ## no corresponding session 0 was detected
                session_0_flag.append('Not started with session 0')
                session = int(df_test['session'][i])
                if session ==1:
                    aligned_flag.append(-session)
                    status_aligned.append(df_test['status'][i])
                elif(session ==2):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    if (len(x1)>0):
                        aligned_flag.append(-1)
                        status_aligned.append(-x1.item())
                    elif len(x1)==0:
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])
                elif(session ==3):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x2 = df_test.loc[(df_test['session']=='2') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    if len(x1)>0:
                        aligned_flag.append(-1)
                        status_aligned.append(x1.item())
                    elif ((len(x1)==0) & (len(x2)>0)):
                        aligned_flag.append(-2)
                        status_aligned.append(-x2.item())
                    elif ((len(x1)==0) & (len(x2)==0)):
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])
                elif(session ==4):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x2 = df_test.loc[(df_test['session']=='2') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x3 = df_test.loc[(df_test['session']=='3') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']

                    if len(x1)>0:
                        aligned_flag.append(-1)
                        status_aligned.append(-x1.item())
                    elif ((len(x1)==0) & (len(x2)>0)):
                        aligned_flag.append(-2)
                        status_aligned.append(x2.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)>0)):
                        aligned_flag.append(-3)
                        status_aligned.append(-x3.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0)):
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])
                elif(session ==5):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x2 = df_test.loc[(df_test['session']=='2') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x3 = df_test.loc[(df_test['session']=='3') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x4 = df_test.loc[(df_test['session']=='4') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']

                    if len(x1)>0:
                        aligned_flag.append(-1)
                        status_aligned.append(x1.item())
                    elif ((len(x1)==0) & (len(x2)>0)):
                        aligned_flag.append(-2)
                        status_aligned.append(-x2.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)>0)):
                        aligned_flag.append(-3)
                        status_aligned.append(x3.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)>0)):
                        aligned_flag.append(-4)
                        status_aligned.append(-x4.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0)):
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])
                elif(session ==6):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x2 = df_test.loc[(df_test['session']=='2') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x3 = df_test.loc[(df_test['session']=='3') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x4 = df_test.loc[(df_test['session']=='4') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x5 = df_test.loc[(df_test['session']=='5') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']

                    if len(x1)>0:
                        aligned_flag.append(-1)
                        status_aligned.append(-x1.item())
                    elif ((len(x1)==0) & (len(x2)>0)):
                        aligned_flag.append(-2)
                        status_aligned.append(x2.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)>0)):
                        aligned_flag.append(-3)
                        status_aligned.append(-x3.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)>0)):
                        aligned_flag.append(-4)
                        status_aligned.append(x4.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0) & (len(x5)>0)):
                        aligned_flag.append(-5)
                        status_aligned.append(-x5.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0) & (len(x5)==0)):
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])

                elif(session ==7):
                    x1 = df_test.loc[(df_test['session']=='1') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x2 = df_test.loc[(df_test['session']=='2') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x3 = df_test.loc[(df_test['session']=='3') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x4 = df_test.loc[(df_test['session']=='4') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x5 = df_test.loc[(df_test['session']=='5') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']
                    x6 = df_test.loc[(df_test['session']=='6') & (df_test['date']==df_test['date'][i])& (df_test['tag']==df_test['tag'][i]) & (df_test['chip_id']==df_test['chip_id'][i])]['status']

                    if len(x1)>0:
                        aligned_flag.append(-1)
                        status_aligned.append(x1.item())
                    elif ((len(x1)==0) & (len(x2)>0)):
                        aligned_flag.append(-2)
                        status_aligned.append(-x2.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)>0)):
                        aligned_flag.append(-3)
                        status_aligned.append(x3.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)>0)):
                        aligned_flag.append(-4)
                        status_aligned.append(-x4.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0) & (len(x5)>0)):
                        aligned_flag.append(-5)
                        status_aligned.append(x5.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0) & (len(x5)==0) & (len(x6)>0)):
                        aligned_flag.append(-6) 
                        status_aligned.append(-x6.item())
                    elif ((len(x1)==0) & (len(x2)==0) & (len(x3)==0) & (len(x4)==0) & (len(x5)==0) & (len(x6)==0)):
                        aligned_flag.append(-session)
                        status_aligned.append(df_test['status'][i])
                else: #not sure if works
                    aligned_flag.append(-session)
                    status_aligned.append(df_test['status'][i])

    df_test['status_aligned'] = status_aligned
    df_test['aligned_session_num'] = aligned_flag
    df_test['start_condition'] = session_0_flag
    df_test['status_real']=99
    for i in range(len(df_test)):
        if (df_test['status_aligned'][i]==-1):
            df_test['status_real'][i]= 'rest'
        else:
            df_test['status_real'][i]= 'full game'
            
    return df_test