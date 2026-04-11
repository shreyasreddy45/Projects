class JoinedStudentModel {
  final String uid;
  final String name;
  final String role;
  final String profilePhotoUrl;

  JoinedStudentModel({
    required this.uid,
    required this.name,
    required this.role,
    required this.profilePhotoUrl,
  });

  factory JoinedStudentModel.fromMap(Map<String, dynamic> data) {
    return JoinedStudentModel(
      uid: data['uid'] ?? '',
      name: data['name'] ?? '',
      role: data['role'] ?? '',
      profilePhotoUrl: data['profilePhotoUrl'] ?? '',
    );
  }
}
