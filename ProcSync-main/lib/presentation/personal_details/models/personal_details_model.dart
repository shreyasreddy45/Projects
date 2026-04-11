class PersonalDetailsModel {
  final String dob;
  final String gender;
  final String bloodGroup;
  final String altPhone;
  final String personalEmail;
  final String aadhar;
  final Map<String, String> presentAddress;
  final Map<String, String> permanentAddress;

  PersonalDetailsModel({
    required this.dob,
    required this.gender,
    required this.bloodGroup,
    required this.altPhone,
    required this.personalEmail,
    required this.aadhar,
    required this.presentAddress,
    required this.permanentAddress,
  });

  Map<String, dynamic> toMap() {
    return {
      'dob': dob,
      'gender': gender,
      'bloodGroup': bloodGroup,
      'altPhone': altPhone,
      'personalEmail': personalEmail,
      'aadhar': aadhar,
      'presentAddress': presentAddress,
      'permanentAddress': permanentAddress,
    };
  }

  factory PersonalDetailsModel.fromMap(Map<String, dynamic> map) {
    return PersonalDetailsModel(
      dob: map['dob'] ?? '',
      gender: map['gender'] ?? '',
      bloodGroup: map['bloodGroup'] ?? '',
      altPhone: map['altPhone'] ?? '',
      personalEmail: map['personalEmail'] ?? '',
      aadhar: map['aadhar'] ?? '',
      presentAddress: Map<String, String>.from(map['presentAddress'] ?? {}),
      permanentAddress: Map<String, String>.from(map['permanentAddress'] ?? {}),
    );
  }
}
