import 'package:flutter_bloc/flutter_bloc.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'academic_details_event.dart';
import 'academic_details_state.dart';

class AcademicDetailsBloc
    extends Bloc<AcademicDetailsEvent, AcademicDetailsState> {
  AcademicDetailsBloc() : super(AcademicDetailsInitial()) {
    on<SubmitAcademicDetails>(_onSubmit);
  }

  Future<void> _onSubmit(
    SubmitAcademicDetails event,
    Emitter<AcademicDetailsState> emit,
  ) async {
    emit(AcademicDetailsSubmitting());
    try {
      final batch = FirebaseFirestore.instance.batch();
      final ref = FirebaseFirestore.instance
          .collection('students')
          .doc(event.userId)
          .collection('info')
          .doc('academic');

      final coursesMap = event.courses.map((e) => e.toMap()).toList();
      batch.set(ref, {'courses': coursesMap});
      await batch.commit();
      emit(AcademicDetailsSuccess());
    } catch (e) {
      emit(AcademicDetailsFailure(e.toString()));
    }
  }
}
