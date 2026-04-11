class FamilyDetailsModel {
  final Map<String, String> father;
  final Map<String, String> mother;

  FamilyDetailsModel({
    required this.father,
    required this.mother,
  });

  Map<String, dynamic> toMap() {
    return {
      'father': father,
      'mother': mother,
    };
  }

  factory FamilyDetailsModel.fromMap(Map<String, dynamic> map) {
    return FamilyDetailsModel(
      father: Map<String, String>.from(map['father'] ?? {}),
      mother: Map<String, String>.from(map['mother'] ?? {}),
    );
  }
}
