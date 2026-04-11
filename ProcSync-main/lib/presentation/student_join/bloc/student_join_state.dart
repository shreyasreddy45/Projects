part of 'student_join_bloc.dart';

enum JoinStatus { initial, loading, success, failure }

class StudentJoinState extends Equatable {
  final String classCode;
  final JoinStatus status;
  final String? errorMessage;
  final UserModel? userModel;
  final ClassModel? classModel; // ✅ New field

  const StudentJoinState({
    this.classCode = '',
    this.status = JoinStatus.initial,
    this.errorMessage,
    this.userModel,
    this.classModel, // ✅ Initialize new field
  });

  StudentJoinState copyWith({
    String? classCode,
    JoinStatus? status,
    String? errorMessage,
    UserModel? userModel,
    ClassModel? classModel, // ✅ Support copying new field
  }) {
    return StudentJoinState(
      classCode: classCode ?? this.classCode,
      status: status ?? this.status,
      errorMessage: errorMessage,
      userModel: userModel ?? this.userModel,
      classModel: classModel ?? this.classModel, // ✅ Copy classModel
    );
  }

  @override
  List<Object?> get props => [
        classCode,
        status,
        errorMessage,
        userModel,
        classModel, // ✅ Ensure it's part of equality check
      ];
}
