part of 'student_join_bloc.dart';

abstract class StudentJoinEvent extends Equatable {
  const StudentJoinEvent();

  @override
  List<Object> get props => [];
}

class ClassCodeChanged extends StudentJoinEvent {
  final String code;

  const ClassCodeChanged(this.code);

  @override
  List<Object> get props => [code];
}

class JoinClassSubmitted extends StudentJoinEvent {}
