part of 'class_creation_bloc.dart';

abstract class ClassCreationEvent {}

class CreateClassEvent extends ClassCreationEvent {
  final String name;
  final String section;
  final String teacherUid;

  CreateClassEvent({
    required this.name,
    required this.section,
    required this.teacherUid,
  });
}
