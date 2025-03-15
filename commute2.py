import numpy as np
import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime, timedelta

def clean_period(period):
    """'기간' 값의 앞 10자리에서 '.'을 제거한 문자열 반환"""
    return re.sub(r'\.', '', str(period)[:10]) #연차,반차의 '기간' 기록과 출퇴근기록 리스트를 병합할때 사용

def round_up_5min(time_input, return_as_str=True):
    """5분 단위로 올림한 시간을 반환 (출근)
    
    Args:
        time_input (str | datetime.time): "HH:MM" 형식의 문자열 또는 datetime.time 객체
        return_as_str (bool): True면 "HH:MM" 문자열 반환, False면 datetime.time 반환
    
    Returns:
        str | datetime.time: 5분 단위 올림한 시간
    """
    if not time_input:
        return None

    try:
        # 문자열 입력일 경우 datetime 객체로 변환
        if isinstance(time_input, str):
            time_obj = datetime.strptime(time_input.zfill(5), "%H:%M")
        elif isinstance(time_input, datetime.time):
            time_obj = datetime.combine(datetime.today(), time_input)  # 날짜 정보 추가
        else:
            return None  # 지원되지 않는 형식

        # 현재 분(minute)을 5분 단위로 올림
        minutes_to_add = (5 - time_obj.minute % 5) % 5  # 올림해야 할 분 계산
        rounded_time = time_obj + timedelta(minutes=minutes_to_add)

        # 반환 타입 결정
        if return_as_str:
            return rounded_time.strftime("%H:%M")  # 문자열 반환
        else:
            return rounded_time.time()  # datetime.time 객체 반환

    except ValueError:
        return None  # 잘못된 시간 형식 처리

def round_down_5min(time_input, return_as_str=True):
    """5분 단위로 '내림'하여 시간 반환 (퇴근)
    
    Args:
        time_input (str | datetime.time): "HH:MM" 형식의 문자열 또는 datetime.time 객체
        return_as_str (bool): True면 "HH:MM" 문자열 반환, False면 datetime.time 반환
    
    Returns:
        str | datetime.time: 5분 단위 내림한 시간
    """
    if not time_input:
        return None

    try:
        # 문자열 입력일 경우 datetime 객체로 변환
        if isinstance(time_input, str):
            time_obj = datetime.strptime(time_input.zfill(5), "%H:%M")
        elif isinstance(time_input, datetime.time):
            time_obj = datetime.combine(datetime.today(), time_input)  # 날짜 정보 추가
        else:
            return None  # 지원되지 않는 형식

        # 현재 분(minute)을 5분 단위로 내림
        minutes_to_subtract = time_obj.minute % 5  # 내림해야 할 분 계산
        rounded_time = time_obj - timedelta(minutes=minutes_to_subtract)

        # 반환 타입 결정
        if return_as_str:
            return rounded_time.strftime("%H:%M")  # 문자열 반환
        else:
            return rounded_time.time()  # datetime.time 객체 반환

    except ValueError:
        return None  # 잘못된 시간 형식 처리

def extract_time(text, round_function=None, return_as_time=False):
    """출근 및 퇴근 열에서 시간을 추출하고, hh:mm 형식으로 변환
       return_as_time=True이면 datetime.time 객체를 반환하여 연산 가능
    """
    if not text:
        return None

    try:
        time_str = str(text)[:5]  # HH:MM 추출
        rounded_time_str = round_function(time_str) if round_function else time_str  # 반올림 적용

        # return_as_time=True이면 datetime.time 객체 반환
        if return_as_time:
            return datetime.strptime(rounded_time_str, "%H:%M").time()
        return rounded_time_str  # 기본적으로 문자열 반환

    except ValueError:
        return None

def determine_working_hours(standard_date):
    """'기준일' 열에서 평일 여부 확인 후 '소정근로시간' 값을 반환 (평일: 08:00:00, 주말: None)"""
    match = re.search(r'(\d{8})\((.)\)', str(standard_date))  
    if match:
        date_str, day_char = match.groups()
        if day_char in "월화수목금":
            return pd.to_timedelta("08:00:00")  # ✅ 시간 형식 변경
    return None

def calculate_actual_work_hours(row):
    """'실제근로시간' 계산 - '일수' 값이 있으면 8시간 × 일수 추가"""
    additional_hours = pd.Timedelta(hours=8 * row['일수']) if pd.notna(row['일수']) else pd.Timedelta(0)

    if pd.isna(row['퇴근1']):  # '퇴근1' 값이 없으면 공란
        return additional_hours if additional_hours > pd.Timedelta(0) else None

    try:
        start_time = datetime.strptime(str(row['출근1']), "%H:%M")
        end_time = datetime.strptime(str(row['퇴근1']), "%H:%M")
        break_time = pd.to_timedelta(row['휴게1']) if pd.notna(row['휴게1']) else pd.Timedelta(0)

        actual_work_duration = end_time - start_time - break_time
        total_work_time = actual_work_duration + additional_hours

        return pd.to_timedelta(total_work_time) if total_work_time.total_seconds() >= 0 else None
    except (ValueError, TypeError):
        return None


def calculate_break_time(row):
    """출근1 및 퇴근1 시간이 점심시간(12:00~13:00)의 80% 이상을 포함하면 '01:00', 그렇지 않으면 '00:00' 반환"""
    
    # 출근 또는 퇴근 값이 없으면 휴게시간 없음
    if pd.isna(row['출근1']) or pd.isna(row['퇴근1']):
        return pd.Timedelta(0)

    try:
        # 출근/퇴근 시간을 datetime 객체로 변환
        start_time = datetime.strptime(str(row['출근1']), "%H:%M")
        end_time = datetime.strptime(str(row['퇴근1']), "%H:%M")

        # 점심시간 범위 (12:00 ~ 13:00)
        lunch_start = datetime.strptime("12:00", "%H:%M")
        lunch_end = datetime.strptime("13:00", "%H:%M")

        # 출근/퇴근이 점심시간을 포함하는 경우 계산
        if end_time <= lunch_start or start_time >= lunch_end:
            return pd.Timedelta(0)  # 점심시간 포함 안됨

        # 점심시간 내에서 실제 근무한 시간 계산
        lunch_overlap_start = max(start_time, lunch_start)
        lunch_overlap_end = min(end_time, lunch_end)
        lunch_overlap_duration = lunch_overlap_end - lunch_overlap_start  # timedelta 계산

        # 점심시간 60분 중 포함된 비율 계산
        lunch_percentage = lunch_overlap_duration.total_seconds() / 3600  # 시간 단위
        if lunch_percentage >= 0.8:  # 80% 이상 포함
            return pd.Timedelta(hours=1)  # 01:00:00 반환
        else:
            return pd.Timedelta(0)  # 00:00:00 반환

    except ValueError:
        return pd.Timedelta(0)  # 오류 발생 시 기본값 반환


def merge_data(commute_df, absence_df):
    """이 함수는 commute_df(기본 근태 데이터)와 absence_df(추가 결근 데이터)를 병합하여, 근무 시간을 계산하고 필요한 컬럼을 추가하는 역할을 한다."""
    """[기본 자료]와 [추가 자료]를 병합하고 '출근1', '퇴근1', '소정근로시간', '휴게1', '실제근로시간' 컬럼 추가"""
    absence_df['기간_정리'] = absence_df['기간'].astype(str).apply(clean_period) #absence_df['기간']: 날짜를 "YYYYMMDD" 형식으로 정리 (clean_period() 함수 사용)
    commute_df['기준일_정리'] = commute_df['기준일'].astype(str).str[:8] #commute_df['기준일']: 앞 8자리(YYYYMMDD)만 추출하여 정리

    # 출근1 (5분 단위 올림) 및 퇴근1 (5분 단위 내림) 컬럼 추가
    commute_df['출근1'] = commute_df['출근'].apply(lambda x: extract_time(x, round_up_5min)) #"출근1": 출근 컬럼에서 시간 추출 후 5분 단위 올림 (round_up_5min())
    commute_df['퇴근1'] = commute_df['퇴근'].apply(lambda x: extract_time(x, round_down_5min)) #"퇴근1": 퇴근 컬럼에서 시간 추출 후 5분 단위 내림 (round_down_5min())

    # 소정근로시간 컬럼 추가 (평일에만 08:00, 주말은 NaN)
    commute_df['소정근로시간'] = commute_df['기준일'].apply(determine_working_hours)

    # 휴게1 컬럼 추가 ('출근1'이 있으면 '01:00:00', 없으면 NaN)
    commute_df['휴게1'] = commute_df.apply(calculate_break_time, axis=1)

    # '일수' 컬럼을 숫자로 변환 (NaN 값 방지)
    absence_df['일수'] = pd.to_numeric(absence_df['일수'], errors='coerce')

    # 병합 수행 (이름과 기준일을 기준으로)
    merged_df = commute_df.merge(
        absence_df[['이름', '기간_정리', '일수']],
        left_on=['이름', '기준일_정리'],
        right_on=['이름', '기간_정리'],
        how='left'
    )

    # 실제근로시간 계산
    merged_df['실제근로시간'] = merged_df.apply(calculate_actual_work_hours, axis=1)

    # 불필요한 컬럼 제거 후 정리
    merged_df.drop(columns=['기간_정리', '기준일_정리'], inplace=True)

    return merged_df

def sum_actual_work_hours(df):
    """직원별 근로시간 합산 후, 초과근무시간 및 휴일근무시간 계산"""
    
    # ✅ timedelta 변환 (연산 가능하도록)
    df['실제근로시간'] = pd.to_timedelta(df['실제근로시간'], errors='coerce')
    df['소정근로시간'] = pd.to_timedelta(df['소정근로시간'], errors='coerce')

    # ✅ '휴일여부' 추가 (기준일이 '(일)'인지 여부 확인)
    df['휴일여부'] = df['기준일'].astype(str).str[-3:] == "(일)"

    # ✅ 직원별 근로시간 합산
    summary_df = df.groupby('이름')[['실제근로시간', '소정근로시간']].sum().reset_index()

    # ✅ 휴일근무시간 계산 (기준일이 '(일)'인 경우의 실제근로시간 합산)
    holiday_work_df = df[df['휴일여부']].groupby('이름')['실제근로시간'].sum().reset_index()
    holiday_work_df.rename(columns={'실제근로시간': '휴일근무시간'}, inplace=True)

    # ✅ summary_df에 병합 (휴일근무시간 추가)
    summary_df = summary_df.merge(holiday_work_df, on="이름", how="left")
    summary_df['휴일근무시간'] = summary_df['휴일근무시간'].fillna(pd.Timedelta(0))  # NaN을 0으로 처리

    # ✅ 초과근무시간 계산
    summary_df['초과근무시간'] = summary_df['실제근로시간'] - summary_df['소정근로시간']

    # ✅ 초과근무시간에서 휴일근무시간 제외
    summary_df['초과근무시간Re'] = summary_df['초과근무시간'] - summary_df['휴일근무시간']

    return summary_df

def merge_ordinary_hourly_wage(summary_df, wage_df):
    """'통상시급' 데이터와 summary_df를 병합"""
    
    # 필요한 컬럼만 선택
    wage_df = wage_df[['이름', '통상시급']]
    
    # 데이터 병합 (왼쪽 기준)
    merged_summary = summary_df.merge(wage_df, on='이름', how='left')

    # NaN 처리 (없는 직원의 통상시급을 0으로 설정)
    merged_summary['통상시급'] = merged_summary['통상시급'].fillna(0).astype(int)

    return merged_summary

def calculate_overtime_pay(summary_df):
    """초과근무수당을 계산하여 summary_df에 추가"""
   
    # ✅ 초과근무시간을 timedelta → 초 단위로 변환 후, 시간 단위로 변환 (엑셀과 동일한 변환 방식)
    if np.issubdtype(summary_df['초과근무시간Re'].dtype, np.timedelta64):
        summary_df['초과근무시간_변환'] = summary_df['초과근무시간Re'].dt.total_seconds() / 3600
    else:
        summary_df['초과근무시간_변환'] = summary_df['초과근무시간Re'].astype(float) * 24  

    # 초과근무수당 계산 (변환된 초과근무시간Re × 통상시급 × 1.5)
    summary_df['초과근무수당'] = summary_df['초과근무시간_변환'] * summary_df['통상시급'] * 1.5

    # NaN 값 제거 및 float → int 변환
    summary_df['초과근무수당'] = summary_df['초과근무수당'].fillna(0).round(0).astype(int, errors='ignore')
     
    # 휴일근무수당 계산
    if np.issubdtype(summary_df['휴일근무시간'].dtype, np.timedelta64):
        summary_df['휴일근무시간_변환'] = summary_df['휴일근무시간'].dt.total_seconds() / 3600
    else:
        summary_df['휴일근무시간_변환'] = summary_df['휴일근무시간'].astype(float) * 24 

    # 휴일근무수당 계산 (변환된 휴일근무시간 × 통상시급 × 2)
    summary_df['휴일근무수당'] = summary_df['휴일근무시간_변환'] * summary_df['통상시급'] * 2

    # NaN 값 제거 및 float → int 변환
    summary_df['휴일근무수당'] = summary_df['휴일근무수당'].fillna(0).round(0).astype(int, errors='ignore')

    # ✅ '초과근무시간_변환', '휴일근무시간_변환' 컬럼 제거 (엑셀 파일에서 표시되지 않도록 함)
    summary_df = summary_df.drop(columns=['초과근무시간_변환', '휴일근무시간_변환'])

    # 연장근로수당 합계: '초과근무수당' + '휴일근무수당'
    summary_df['연장근로수당'] = summary_df['초과근무수당'] + summary_df['휴일근무수당']

    return summary_df

st.title('엑셀 데이터 병합')

# 기본 자료 업로드
commute_file = st.file_uploader("[기본 자료] commuteList 엑셀 파일 업로드", type=['xlsx'])
if commute_file:
    if 'commuteList' not in commute_file.name:
        st.error("파일명이 'commuteList'를 포함해야 합니다.")
    else:
        commute_df = pd.read_excel(commute_file)
        st.success("[기본 자료] 파일 업로드 성공")

        # 추가 자료 업로드
        absence_file = st.file_uploader("[추가 자료] absenceTimeOffList 엑셀 파일 업로드", type=['xlsx'])
        if absence_file:
            if 'absenceTimeOffList' not in absence_file.name:
                st.error("파일명이 'absenceTimeOffList'를 포함해야 합니다.")
            else:
                absence_df = pd.read_excel(absence_file)
                st.success("[추가 자료] 파일 업로드 성공")

                # ✅ 데이터 병합
                result_df = merge_data(commute_df, absence_df)

                # ✅ 직원별 총 실제 근로시간 계산
                summary_df = sum_actual_work_hours(result_df)
                
                # ✅ 추가 시급 데이터 업로드
                wage_file = st.file_uploader("[추가 시급 자료] ordinary_hourly_wage 엑셀 파일 업로드", type=['xlsx'])
                if wage_file:
                    if 'ordinary_hourly_wage' not in wage_file.name:
                        st.error("파일명이 'ordinary_hourly_wage'를 포함해야 합니다.")
                    else:
                        wage_df = pd.read_excel(wage_file)
                        st.success("[추가 시급 자료] 파일 업로드 성공")
                        
                        # ✅ 통상상시급 데이터 병합
                        summary_df = merge_ordinary_hourly_wage(summary_df, wage_df)
                        
                        # ✅ 초과근무수당 계산
                        summary_df = calculate_overtime_pay(summary_df)


                # ✅ 엑셀 저장 및 다운로드
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for name, group in result_df.groupby('이름'): # result_df.groupby('이름')을 이용해 각 직원별 데이터를 개별 시트로 저장.
                        group.to_excel(writer, index=False, sheet_name=str(name))

                    # ✅ Summary 시트 추가 (비어있지 않은 경우에만 추가)
                    if not summary_df.empty:
                        summary_df.to_excel(writer, index=False, sheet_name="Summary")

                        if "Summary" in writer.sheets:
                            summary_worksheet = writer.sheets["Summary"]

                # ✅ 파일을 메모리 버퍼에 저장한 후에 포인터를 이동
                output.seek(0)  

                # ✅ 파일 크기 확인
                file_size = output.getbuffer().nbytes
                st.write(f"파일 크기: {file_size} bytes")

                if file_size == 0:
                    st.error("파일이 정상적으로 생성되지 않았습니다. 데이터를 확인해주세요.")
                    st.stop()  # 실행 중단

                else:
                    st.download_button(
                        label="이름별 병합된 파일 다운로드 (Summary 포함)",
                        data=output,
                        file_name="merged_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )


