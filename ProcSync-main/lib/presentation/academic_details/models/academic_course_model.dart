class AcademicCourseModel {
  final String courseTitle;
  final String courseCode;
  final String credits;
  final String registerType;
  final String attempts;
  final String faculty;
  final String status;
  final String attendance;
  final String cie;
  final String see;
  final String changeInGrade;
  final String gradePoints;

  AcademicCourseModel({
    required this.courseTitle,
    required this.courseCode,
    required this.credits,
    required this.registerType,
    required this.attempts,
    required this.faculty,
    required this.status,
    required this.attendance,
    required this.cie,
    required this.see,
    required this.changeInGrade,
    required this.gradePoints,
  });

  Map<String, dynamic> toMap() => {
        'courseTitle': courseTitle,
        'courseCode': courseCode,
        'credits': credits,
        'registerType': registerType,
        'attempts': attempts,
        'faculty': faculty,
        'status': status,
        'attendance': attendance,
        'cie': cie,
        'see': see,
        'changeInGrade': changeInGrade,
        'gradePoints': gradePoints,
      };

  factory AcademicCourseModel.fromMap(Map<String, dynamic> map) =>
      AcademicCourseModel(
        courseTitle: map['courseTitle'] ?? '',
        courseCode: map['courseCode'] ?? '',
        credits: map['credits'] ?? '',
        registerType: map['registerType'] ?? '',
        attempts: map['attempts'] ?? '',
        faculty: map['faculty'] ?? '',
        status: map['status'] ?? '',
        attendance: map['attendance'] ?? '',
        cie: map['cie'] ?? '',
        see: map['see'] ?? '',
        changeInGrade: map['changeInGrade'] ?? '',
        gradePoints: map['gradePoints'] ?? '',
      );
}
