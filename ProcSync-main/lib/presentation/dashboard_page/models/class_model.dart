import 'package:cloud_firestore/cloud_firestore.dart';

class ClassModel {
  final String id; // Document ID
  final String code;
  final String name;
  final String section;
  final String createdBy;
  final DateTime createdAt;
  final bool archived;

  ClassModel({
    required this.id,
    required this.code,
    required this.name,
    required this.section,
    required this.createdBy,
    required this.createdAt,
    this.archived = false,
  });

  Map<String, dynamic> toMap() {
    return {
      'code': code,
      'name': name,
      'section': section,
      'createdBy': createdBy,
      'createdAt': Timestamp.fromDate(createdAt),
      'archived': archived,
    };
  }

  factory ClassModel.fromMap(Map<String, dynamic> map, {required String id}) {
    return ClassModel(
      id: id,
      code: map['code'] ?? '',
      name: map['name'] ?? '',
      section: map['section'] ?? '',
      createdBy: map['createdBy'] ?? '',
      createdAt: (map['createdAt'] as Timestamp).toDate(),
      archived: map['archived'] ?? false,
    );
  }
}
