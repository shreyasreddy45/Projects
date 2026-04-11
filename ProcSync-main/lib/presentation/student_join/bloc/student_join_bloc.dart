import 'package:bloc/bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:equatable/equatable.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../chat/models/user_model.dart';
import 'package:ProcSync/presentation/dashboard_page/models/class_model.dart';

part 'student_join_event.dart';
part 'student_join_state.dart';

class StudentJoinBloc extends Bloc<StudentJoinEvent, StudentJoinState> {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  StudentJoinBloc() : super(const StudentJoinState()) {
    on<ClassCodeChanged>((event, emit) {
      emit(state.copyWith(classCode: event.code, status: JoinStatus.initial));
    });

    on<JoinClassSubmitted>(_onJoinClass);
  }

  Future<void> _onJoinClass(
    JoinClassSubmitted event,
    Emitter<StudentJoinState> emit,
  ) async {
    final user = _auth.currentUser;
    final code = state.classCode.trim();

    if (user == null || code.isEmpty) {
      emit(state.copyWith(
        status: JoinStatus.failure,
        errorMessage: 'Invalid class code or not signed in.',
      ));
      return;
    }

    if (code.length < 6 || code.length > 8) {
      emit(state.copyWith(
        status: JoinStatus.failure,
        errorMessage: 'Class code should be 6 to 8 characters.',
      ));
      return;
    }

    try {
      emit(state.copyWith(status: JoinStatus.loading));

      // 🔍 Fetch class
      final classDoc =
          await _firestore.collection('classrooms').doc(code).get();

      if (!classDoc.exists) {
        emit(state.copyWith(
          status: JoinStatus.failure,
          errorMessage: "Class not found.",
        ));
        return;
      }

      final classData = classDoc.data();
      if (classData == null) {
        emit(state.copyWith(
          status: JoinStatus.failure,
          errorMessage: "Class data is corrupted.",
        ));
        return;
      }

      // 🧾 Ensure user document exists
      final userRef = _firestore.collection('users').doc(user.uid);
      final userSnap = await userRef.get();
      if (!userSnap.exists) {
        emit(state.copyWith(
          status: JoinStatus.failure,
          errorMessage:
              "User profile not found. Please complete profile setup.",
        ));
        return;
      }

      final userData = userSnap.data()!;
      final userModel = UserModel.fromMap(userData);

      // Check if already joined (class 'students' array)
      final studentsRaw = classData['students'];
      final students = (studentsRaw is List)
          ? studentsRaw.map((e) => e.toString()).toList()
          : <String>[];

      if (students.contains(user.uid)) {
        emit(state.copyWith(
          status: JoinStatus.failure,
          errorMessage: "You’re already in this class.",
        ));
        return;
      }

      // 🔄 Update user's joined class
      await userRef.update({'joinedClassId': code});

      // ➕ Add UID to students array in class
      await _firestore.collection('classrooms').doc(code).update({
        'students': FieldValue.arrayUnion([user.uid]),
      });

      // ✅ Add full student info to classroom > students subcollection
      await _firestore
          .collection('classrooms')
          .doc(code)
          .collection('students')
          .doc(user.uid)
          .set({
        'uid': userModel.uid,
        'name': userModel.name,
        'email': userModel.email,
        'photoUrl': userModel.photoUrl ?? '',
      });

      // ✅ Success: Emit state with updated models
      final classModel = ClassModel.fromMap(classData, id: classDoc.id);

      emit(state.copyWith(
        status: JoinStatus.success,
        userModel: userModel,
        classModel: classModel,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: JoinStatus.failure,
        errorMessage: "Error: ${e.toString()}",
      ));
    }
  }
}
