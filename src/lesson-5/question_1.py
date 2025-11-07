"""
1. Upload Registration.csv Download Registration.csvand Course_info.xlsx Download Course_info.xlsxinto Pandas. (5 points)
2. Explore and clean Registration data. (30 points)
3. Explore and clean Course info data. (10 points)
4. Which course has the highest registration? (15 points)
5. Propose a solution to mitigate the data inconsistency (e.g., naming inconsistency) existing in the two datasets, and perform an inner join on them. (20 points)
6. Create a data frame with student names as the index, course numbers as columns, and if the student registered a course as values (0, 1). (20 points)
"""
import pandas as pd

# 1. Load the dataset
data_registration = pd.read_csv("./data/lesson-5/Registration.csv")
data_course_info = pd.read_csv("./data/lesson-5/Course_info.csv")

# 2. Explore and clean Registration data.
print("First 5 rows of the Registration dataset:")
print(data_registration.head())
print("\nStatistical summary of the Registration dataset:")
print(data_registration.describe())
print("\nCount of each column in Registration dataset:")
print(data_registration.count())

# Check for missing values in Registration data
print("\nMissing values in each column of Registration dataset:")
print(data_registration.isnull().sum())

# Handling missing data
# Remove rows with any missing data
data_registration = data_registration.dropna()
# Handling missing values by filling them with the mode of the column
# for col in data_registration.columns:
#     if data_registration[col].isnull().sum() > 0:
#         data_registration[col] = data_registration[col].fillna(data_registration[col].mode()[0])

print(data_registration.head())
print("\nMissing values in each column of Registration dataset:")
print(data_registration.isnull().sum())

#` 3. Explore and clean Course info data.
print("\nFirst 5 rows of the Course Info dataset:")
print(data_course_info.head())
print("\nStatistical summary of the Course Info dataset:")
print(data_course_info.describe())
print("\nCount of each column in Course Info dataset:")
print(data_course_info.count())
# Check for missing values in Course Info data
print("\nMissing values in each column of Course Info dataset:")
print(data_course_info.isnull().sum())
# Handling missing data
# Remove rows with any missing data
data_course_info = data_course_info.dropna()
print(data_course_info.head())
print("\nMissing values in each column of Course Info dataset:")
print(data_course_info.isnull().sum())

# 4. Which course has the highest registration?
course_registration_counts = data_registration['coursename'].value_counts()
most_registered_course = course_registration_counts.idxmax()
most_registered_count = course_registration_counts.max()
print(f"\nCourse with the highest registration: {most_registered_course} with {most_registered_count} registrations.")

# clean up column names by stripping leading/trailing spaces
data_course_info.columns = data_course_info.columns.str.strip()
#print(data_course_info.columns)

# Find the course number for the most registered course
course_number = None
values = data_course_info.loc[
    data_course_info['Course Name'] == most_registered_course, 'Course number'
    # data_course_info['Course Name'].contains(most_registered_course), 'Course number'
    # data_course_info[data_course_info['Course Name'].str.contains(most_registered_course, case=False, na=False)]['Course number']
]

if not values.empty:
    course_number = values.values[0]

print(f"Course number for the most registered course ({most_registered_course}): {course_number}")

# Propose a solution to mitigate the data inconsistency (e.g., naming inconsistency)
# existing in the two datasets, and perform an inner join on them
# Solution: Standardize course names by converting to lowercase and stripping spaces
data_registration['coursename'] = data_registration['coursename'].str.strip().str.upper()
data_course_info['Course Name'] = data_course_info['Course Name'].str.strip().str.upper()
merged_data = pd.merge(
    data_registration,
    data_course_info,
    left_on='coursename',
    right_on='Course Name',
    how='inner'
)
print("\nMerged Data (inner join on cleaned course names):")
print(merged_data)

print("\nMissing values in each column of the merged dataset:")
print(data_registration.isnull().sum())
# write to a csv file
# merged_data.to_csv("./data/lesson-5/Merged_Registration_CourseInfo.csv", index=False)

# compare the row counts before and after the merge
print(f"\nRow count in Registration data: {data_registration.shape[0]}")
print(f"Row count in Course Info data: {data_course_info.shape[0]}")
print(f"Row count in Merged data: {merged_data.shape[0]}")

#print course names that didn't match
unmatched_courses = set(data_registration['coursename']) - set(data_course_info['Course Name'])
print(f"\nCourses in Registration data that didn't match Course Info data: {unmatched_courses}")

# 6. Create a data frame with student names as the index, course numbers as columns,
# and if the student registered a course as values (0, 1).
pivot_table = pd.pivot_table(
    merged_data,
    index='Student name',
    columns='Course number',
    aggfunc='size',
    fill_value=0
)

print("\nPivot Table with student names as index and course numbers as columns:")
print(pivot_table)





