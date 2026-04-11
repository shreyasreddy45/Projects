import 'package:bloc/bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'student_info_event.dart';
import 'student_info_state.dart';
import '../models/student_info_model.dart';

class StudentInfoBloc extends Bloc<StudentInfoEvent, StudentInfoState> {
  final FirebaseFirestore firestore;
  final FirebaseStorage storage;

  StudentInfoBloc({required this.firestore, required this.storage})
      : super(StudentInfoInitial()) {
    on<LoadStudentInfo>(_onLoadStudentInfo);
    on<UploadProfilePhoto>(_onUploadProfilePhoto);
    on<UpdatePersonalDetails>(_onUpdatePersonalDetails);
    on<UpdateParentDetails>(_onUpdateParentDetails);
    on<UpdateEducationDetails>(_onUpdateEducationDetails);
  }

  Future<void> _onLoadStudentInfo(
      LoadStudentInfo event, Emitter<StudentInfoState> emit) async {
    emit(StudentInfoLoading());
    try {
      final doc = await firestore.collection('students').doc(event.uid).get();
      if (doc.exists) {
        emit(StudentInfoLoaded(StudentInfo.fromMap(doc.data()!)));
      } else {
        emit(StudentInfoLoaded(StudentInfo(
          uid: event.uid,
          profilePhotoUrl: '',
          personalDetails: {},
          parentDetails: {},
          educationalDetails: {},
        )));
      }
    } catch (e) {
      emit(StudentInfoError(e.toString()));
    }
  }

  Future<void> _onUploadProfilePhoto(
      UploadProfilePhoto event, Emitter<StudentInfoState> emit) async {
    emit(StudentInfoLoading());
    try {
      final ref = storage.ref().child('profile_photos/${event.uid}.jpg');
      await ref.putFile(event.imageFile);
      final url = await ref.getDownloadURL();

      final docRef = firestore.collection('students').doc(event.uid);
      await docRef.set({'profilePhotoUrl': url}, SetOptions(merge: true));

      final doc = await docRef.get();
      emit(StudentInfoLoaded(StudentInfo.fromMap(doc.data()!)));
    } catch (e) {
      emit(StudentInfoError(e.toString()));
    }
  }

  Future<void> _updateSection(
    String uid,
    Map<String, dynamic> data,
    String section,
    Emitter<StudentInfoState> emit,
  ) async {
    emit(StudentInfoLoading());
    try {
      final docRef = firestore.collection('students').doc(uid);
      await docRef.set({section: data}, SetOptions(merge: true));
      final doc = await docRef.get();
      emit(StudentInfoLoaded(StudentInfo.fromMap(doc.data()!)));
    } catch (e) {
      emit(StudentInfoError(e.toString()));
    }
  }

  Future<void> _onUpdatePersonalDetails(
      UpdatePersonalDetails event, Emitter<StudentInfoState> emit) async {
    await _updateSection(
        event.uid, event.personalDetails, 'personalDetails', emit);
  }

  Future<void> _onUpdateParentDetails(
      UpdateParentDetails event, Emitter<StudentInfoState> emit) async {
    await _updateSection(event.uid, event.parentDetails, 'parentDetails', emit);
  }

  Future<void> _onUpdateEducationDetails(
      UpdateEducationDetails event, Emitter<StudentInfoState> emit) async {
    await _updateSection(
        event.uid, event.educationDetails, 'educationDetails', emit);
  }
}
