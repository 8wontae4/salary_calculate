import streamlit as st
import pandas as pd
import io
import re
from datetime import datetime, timedelta

def clean_period(period):
    """'기간' 값의 앞 10자리에서 '.'을 제거한 문자열 반환"""
    return re.sub(r'\.', '', str(period)[:10])

def round_up_5min(time_str):
    """5분 단위로 올림하여 반올림된 시간 반환 (출근)"""
    if not time_str:
        return None
    try:
        time_obj = datetime.strptime(time_str, "%H:%M")
        minute = (time_obj.minute + 4) // 5 * 5  # 5분 단위 올림
        if minute == 60:  # 올림 후 60분이 되면 한 시간 추가
            time_obj = time_obj.replace(hour=(time_obj.hour + 1) % 24, minute=0)
        else:
            time_obj = time_obj.replace(minute=minute)
        return time_obj.time()
    except ValueError:
        return None

def round_down_5min(time_str):
    """5분 단위로 내림하여 반올림된 시간 반환 (퇴근)"""
    if not time_str:
        return None
    try:
        time_obj = datetime.strptime(time_str, "%H:%M")
        minute = time_obj.minute // 5 * 5  # 5분 단위 내림
        time_obj = time_obj.replace(minute=minute)
        return time_obj.time()
    except ValueError:
        return None

def extract_time(text, round_function=None):
    """출근 및 퇴근 열에서 시간을 추출하고, hh:mm:ss 형식으로 변환"""
    time_str = str(text)[:5]  # 앞 5자리만 추출 (HH:MM)
    time_result = round_function(time_str) if round_function else None
    return time_result

def determine_working_hours(standard_date):
    """'기준일' 열에서 평일 여부 확인 후 '소정근로시간' 값을 반환 (평일: 08:00:00, 주말: None)"""
    match = re.search(r'(\d{8})\((.)\)', str(standard_date))  # 날짜 및 요일 추출
    if match:
        date_str, day_char = match.groups()
        if day_char in "월화수목금":  # 평일이면 08:00:00 반환
            return pd.to_timedelta("08:00:00")  # ✅ timedelta로 변환
    return None  # 주말이면 NaN


def calculate_actual_work_hours(row):
    """실제근로시간 계산"""
    if pd.notna(row['일수']):  # '일수' 값이 있는 경우
        hours = float(row['일수']) * 8
        if pd.notna(hours):  # ✅ NaN 체크 추가
            return pd.to_timedelta(f"{int(hours)}:00:00")  # ✅ 시간 정수 변환
        else:
            return None  # NaN이면 None 반환

    if pd.isna(row['퇴근1']):  # '퇴근1' 값이 없으면 공란
        return None

    try:
        start_time = datetime.strptime(str(row['출근1']), "%H:%M:%S")
        end_time = datetime.strptime(str(row['퇴근1']), "%H:%M:%S")
        break_time = datetime.strptime(row['휴게1'], "%H:%M:%S") - datetime.strptime("00:00:00", "%H:%M:%S")

        actual_work_duration = end_time - start_time - break_time
        if actual_work_duration.total_seconds() >= 0:  # ✅ 음수 시간 방지
            return pd.to_timedelta(actual_work_duration)  
        else:
            return None
    except (ValueError, TypeError):
        return None

def merge_data(commute_df, absence_df):
    """[기본 자료]와 [추가 자료]를 병합하고 '출근1', '퇴근1', '소정근로시간', '휴게1', '실제근로시간' 컬럼 추가"""
    absence_df['기간_정리'] = absence_df['기간'].astype(str).apply(clean_period)
    commute_df['기준일_정리'] = commute_df['기준일'].astype(str).str[:8]

    # 출근1 (5분 단위 올림) 및 퇴근1 (5분 단위 내림) 컬럼 추가
    commute_df['출근1'] = commute_df['출근'].apply(lambda x: extract_time(x, round_up_5min))
    commute_df['퇴근1'] = commute_df['퇴근'].apply(lambda x: extract_time(x, round_down_5min))

    # 소정근로시간 컬럼 추가 (평일에만 1, 주말은 NaN)
    commute_df['소정근로시간'] = commute_df['기준일'].apply(determine_working_hours)

    # 휴게1 컬럼 추가 ('출근1'이 있으면 '01:00:00', 없으면 NaN)
    commute_df['휴게1'] = commute_df['출근1'].apply(lambda x: "01:00:00" if pd.notna(x) else None)

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
    """직원별 근로시간 합산 후, 엑셀 연산 가능 데이터 포함"""
    df['실제근로시간'] = pd.to_timedelta(df['실제근로시간'], errors='coerce')  # timedelta 변환
    df['소정근로시간'] = pd.to_timedelta(df['소정근로시간'], errors='coerce')  # timedelta 변환
    
    # ✅ '휴일근무시간' 계산을 위해 '기준일'에서 (일) 여부 추출
    df['휴일여부'] = df['기준일'].astype(str).str[-3:] == "(일)"
    
    # ✅ 직원별 근로시간 계산
    summary_df = df.groupby('이름')[['실제근로시간', '소정근로시간']].sum().reset_index()
    
    # ✅ 휴일근무시간 계산 (기준일이 '(일)'인 경우의 실제근로시간 합산)
    holiday_work_df = df[df['휴일여부']].groupby('이름')['실제근로시간'].sum().reset_index()
    holiday_work_df.rename(columns={'실제근로시간': '휴일근무시간'}, inplace=True)
    
    # ✅ summary_df에 병합
    summary_df = summary_df.merge(holiday_work_df, on="이름", how="left")
    summary_df['휴일근무시간'] = summary_df['휴일근무시간'].fillna(pd.Timedelta(0))  # NaN을 0으로 처리
    
    # ✅ 초과근무시간 계산
    summary_df['초과근무시간'] = summary_df['실제근로시간'] - summary_df['소정근로시간']

    # ✅ 엑셀 연산 가능하도록 float 변환 (시간 단위)
    def timedelta_to_hours(td):
        return td.total_seconds() / 3600 if pd.notna(td) else 0

    summary_df['실제근로시간1'] = summary_df['실제근로시간'].apply(timedelta_to_hours)
    summary_df['소정근로시간1'] = summary_df['소정근로시간'].apply(timedelta_to_hours)
    summary_df['휴일근무시간1'] = summary_df['휴일근무시간'].apply(timedelta_to_hours)
    summary_df['초과근무시간1'] = summary_df['초과근무시간'].apply(timedelta_to_hours)

    # ✅ '초과근무시간Re' 계산 (초과근무시간1 - 휴일근무시간1)
    summary_df['초과근무시간Re'] = summary_df['초과근무시간1'] - summary_df['휴일근무시간1']

    # ✅ "X시간 Y분" 형식 변환 함수
    def format_timedelta(td):
        if pd.isna(td):
            return None
        total_minutes = int(td.total_seconds() // 60)
        hours, minutes = divmod(total_minutes, 60)
        return f"{hours}시간 {minutes}분"

    # ✅ 음수 값 절대값 처리 및 '()' 씌우기
    def format_overtime(td):
        if pd.isna(td):
            return None
        total_minutes = int(abs(td.total_seconds()) // 60)
        hours, minutes = divmod(total_minutes, 60)
        formatted_time = f"{hours}시간 {minutes}분"
        return f"({formatted_time})" if td.total_seconds() < 0 else formatted_time

    summary_df['실제근로시간'] = summary_df['실제근로시간'].apply(format_timedelta)
    summary_df['소정근로시간'] = summary_df['소정근로시간'].apply(format_timedelta)
    summary_df['초과근무시간'] = summary_df['초과근무시간'].apply(format_overtime)
    summary_df['휴일근무시간'] = summary_df['휴일근무시간'].apply(format_timedelta)

    return summary_df


st.title('엑셀 데이터 병합')

# 기본 자료 업로드
commute_file = st.file_uploader("[기본 자료] 엑셀 파일 업로드", type=['xlsx'])
if commute_file:
    if 'commuteList' not in commute_file.name:
        st.error("파일명이 'commuteList'를 포함해야 합니다.")
    else:
        commute_df = pd.read_excel(commute_file)
        st.success("[기본 자료] 파일 업로드 성공")

        # 추가 자료 업로드
        absence_file = st.file_uploader("[추가 자료] 엑셀 파일 업로드", type=['xlsx'])
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

                # ✅ 엑셀 저장 및 다운로드
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    for name, group in result_df.groupby('이름'):
                        group.to_excel(writer, index=False, sheet_name=str(name))

                        # ✅ '소정근로시간' & '실제근로시간' 컬럼을 Excel 시간 서식으로 지정
                        workbook = writer.book  
                        time_format = workbook.add_format({'num_format': 'hh:mm:ss'})  

                        worksheet = writer.sheets[str(name)]
                        
                        if "소정근로시간" in group.columns:
                            col_idx_fixed = group.columns.get_loc("소정근로시간") + 1  
                            worksheet.set_column(col_idx_fixed, col_idx_fixed, None, time_format)

                        if "실제근로시간" in group.columns:
                            col_idx_actual = group.columns.get_loc("실제근로시간") + 1
                            worksheet.set_column(col_idx_actual, col_idx_actual, None, time_format)

                    # ✅ Summary 시트 추가 (비어있지 않은 경우에만 추가)
                    # ✅ Summary 시트 추가 (비어있지 않은 경우에만 추가)
                    if not summary_df.empty:
                        summary_df.to_excel(writer, index=False, sheet_name="Summary")

                        if "Summary" in writer.sheets:
                            summary_worksheet = writer.sheets["Summary"]
                            
                            # ✅ '실제근로시간', '소정근로시간', '초과근무시간', '휴일근무시간' 컬럼 서식 적용
                            time_columns = ["실제근로시간", "소정근로시간", "초과근무시간", "휴일근무시간"]
                            for col in time_columns:
                                if col in summary_df.columns:
                                    col_idx = summary_df.columns.get_loc(col) + 1
                                    summary_worksheet.set_column(col_idx, col_idx, None)

                            # ✅ 연산 가능 컬럼 ('실제근로시간1', '소정근로시간1', '초과근무시간1', '휴일근무시간1', '초과근무시간Re') 적용
                            calc_columns = ["실제근로시간1", "소정근로시간1", "초과근무시간1", "휴일근무시간1", "초과근무시간Re"]
                            for col in calc_columns:
                                if col in summary_df.columns:
                                    col_idx = summary_df.columns.get_loc(col) + 1
                                    summary_worksheet.set_column(col_idx, col_idx, None, writer.book.add_format({'num_format': '0.00'}))

                # ✅ 파일을 메모리 버퍼에 저장한 후에 포인터를 이동
                output.seek(0)  

                # ✅ 파일 크기 확인
                file_size = output.getbuffer().nbytes
                st.write(f"파일 크기: {file_size} bytes")

                # ✅ 파일이 0 byte라면 에러 메시지 출력
                if file_size == 0:
                    st.error("파일이 정상적으로 생성되지 않았습니다. 데이터를 확인해주세요.")
                else:
                    st.download_button(
                        label="이름별 병합된 파일 다운로드 (Summary 포함)",
                        data=output,
                        file_name="merged_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )



