class StudentInfo {
  final String uid;
  final String profilePhotoUrl;
  final Map<String, dynamic> personalDetails;
  final Map<String, dynamic> parentDetails;
  final Map<String, dynamic> educationalDetails;

  StudentInfo({
    required this.uid,
    required this.profilePhotoUrl,
    required this.personalDetails,
    required this.parentDetails,
    required this.educationalDetails,
  });

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'profilePhotoUrl': profilePhotoUrl,
      'personalDetails': personalDetails,
      'parentDetails': parentDetails,
      'educationalDetails': educationalDetails,
    };
  }

  factory StudentInfo.fromMap(Map<String, dynamic> map) {
    return StudentInfo(
      uid: map['uid'] ?? '',
      profilePhotoUrl: map['profilePhotoUrl'] ?? '',
      personalDetails: Map<String, dynamic>.from(map['personalDetails'] ?? {}),
      parentDetails: Map<String, dynamic>.from(map['parentDetails'] ?? {}),
      educationalDetails:
          Map<String, dynamic>.from(map['educationalDetails'] ?? {}),
    );
  }
}
