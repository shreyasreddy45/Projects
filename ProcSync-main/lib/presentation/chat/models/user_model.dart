class UserModel {
  final String uid;
  final String name;
  final String email;
  final String role;
  final String? joinedClassId;
  final String? photoUrl; // ✅ Add this line

  const UserModel({
    required this.uid,
    required this.name,
    required this.email,
    required this.role,
    this.joinedClassId,
    this.photoUrl, // ✅ Add this
  });

  factory UserModel.fromMap(Map<String, dynamic> map) {
    return UserModel(
      uid: map['uid'] as String? ?? '',
      name: map['name'] as String? ?? '',
      email: map['email'] as String? ?? '',
      role: map['role'] as String? ?? '',
      joinedClassId: map['joinedClassId'] as String?,
      photoUrl: map['photoUrl'] as String?, // ✅ Add this
    );
  }

  Map<String, dynamic> toMap() {
    return {
      'uid': uid,
      'name': name,
      'email': email,
      'role': role,
      'joinedClassId': joinedClassId,
      'photoUrl': photoUrl, // ✅ Add this
    };
  }

  UserModel copyWith({
    String? uid,
    String? name,
    String? email,
    String? role,
    String? joinedClassId,
    String? photoUrl, // ✅ Add this
  }) {
    return UserModel(
      uid: uid ?? this.uid,
      name: name ?? this.name,
      email: email ?? this.email,
      role: role ?? this.role,
      joinedClassId: joinedClassId ?? this.joinedClassId,
      photoUrl: photoUrl ?? this.photoUrl, // ✅ Add this
    );
  }

  @override
  String toString() {
    return 'UserModel(uid: $uid, name: $name, email: $email, role: $role, joinedClassId: $joinedClassId, photoUrl: $photoUrl)';
  }
}
